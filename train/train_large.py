#!/usr/bin/env python3
# train_large.py  –  10 000-row CPU demo ≤ 64 GB RAM
# ═══════════════════════════════════════════════════
CSV_PATH   = "lmd_full.csv"   # one CSV; adjust or make it a glob
MAX_ROWS   = 10_000
SEQ_LEN    = 256
BATCH      = 8
ACC_STEPS  = 8                # logical batch = 64
EPOCHS     = 5
SAVE_EVERY = 2_000
TICK_MS    = 10               # 1 tick = 10 ms  →  1 min = 6 000 ticks

D_MODEL    = 256
N_HEAD     = 4
N_LAYER    = 4
LR         = 3e-4
OUT_DIR    = "ckpt_10k"

# ────────────────────────────────────────────────
import os, sys, json, math, pandas as pd, torch, torch.nn as nn, re
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
os.environ["OMP_NUM_THREADS"] = "1";  torch.set_num_threads(1)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ---------- helpers ---------------------------------------------------------
tick = lambda x: str(int(round(float(x)*1000 / TICK_MS)))

note_pat_secs  = re.compile(
    r"\[NOTE\] \s*\[PITCH:(.+?)\]\s*\[START:(.+?)\]\s*\[END:(.+?)\]\s*\[DURATION:(.+?)\]",
    re.I)
note_pat_ticks = re.compile(
    r"\[NOTE\] \s*\[PITCH:(.+?)\]\s*\[START_T:(.+?)\]\s*\[END_T:(.+?)\]\s*\[DUR_T:(.+?)\]",
    re.I)

def tokenise_note(tok: str):
    """
    Split a NOTE token into atomic tokens that share embeddings.
    Accepts either the _seconds_ or the _tick_ variant.
    """
    m = note_pat_secs.match(tok)
    if m:                                  # original float-seconds version
        p, s, e, d = m.groups()
        return ["[NOTE]", "[PITCH]", p,
                "[START_T]", tick(s),
                "[END_T]",   tick(e),
                "[DUR_T]",   tick(d)]

    m = note_pat_ticks.match(tok)
    if m:                                  # already quantised, just split
        p, s, e, d = m.groups()
        return ["[NOTE]", "[PITCH]", p,
                "[START_T]", s,
                "[END_T]",   e,
                "[DUR_T]",   d]

    # fallback – unknown format, keep whole token
    return [tok]

def explode(js: str):
    out = []
    for t in json.loads(js):
        if t.startswith("[NOTE]"):
            out.extend(tokenise_note(t))
        else:
            out.append(t)
        if len(out) >= SEQ_LEN:
            break
    return out

# ---------- accelerator -----------------------------------------------------
acc = Accelerator(gradient_accumulation_steps=ACC_STEPS)

# ---------- vocab build (rank-0 only) ---------------------------------------
if acc.is_main_process:
    vocab, nrows = {"[PAD]"}, 0
    for chunk in pd.read_csv(CSV_PATH, usecols=["tokens"],
                             chunksize=500, dtype={"tokens":"string"}):
        for js in chunk["tokens"]:
            vocab.update(explode(js));  nrows += 1
            if nrows % 1_000 == 0 or nrows == MAX_ROWS:
                print(f"[vocab] {nrows:,}/{MAX_ROWS:,}", flush=True)
            if nrows >= MAX_ROWS:  break
        if nrows >= MAX_ROWS:      break
    tok2id = {t:i for i,t in enumerate(sorted(vocab))}
else:
    tok2id = None

# broadcast to all ranks (if >1)
if acc.num_processes > 1:
    import torch.distributed as dist
    obj = [tok2id];  dist.broadcast_object_list(obj, 0);  tok2id = obj[0]

PAD_ID,  V = tok2id["[PAD]"], len(tok2id)
if acc.is_main_process:
    print(f"✓ vocab ready – {V:,} tokens\n", flush=True)

# ---------- streaming dataset ----------------------------------------------
class Stream(IterableDataset):
    def __iter__(self):
        seen = 0
        for chunk in pd.read_csv(CSV_PATH, usecols=["tokens"],
                                 chunksize=1000, dtype={"tokens":"string"}):
            for js in chunk["tokens"]:
                if seen >= MAX_ROWS:  return
                ids  = [tok2id[t] for t in explode(js)]
                full = ids + [PAD_ID]*(SEQ_LEN - len(ids))
                seen += 1
                if seen % 1_000 == 0:
                    print(f"[loader] {seen:,}/{MAX_ROWS:,}", flush=True)
                yield (torch.tensor(full[:-1]),
                       torch.tensor(full[1:]))

loader = DataLoader(Stream(), BATCH, num_workers=0, pin_memory=False)

# ---------- tiny GPT --------------------------------------------------------
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL))
        layer    = nn.TransformerEncoderLayer(
                     D_MODEL, N_HEAD, D_MODEL*4, batch_first=True)
        self.tr  = nn.TransformerEncoder(layer, N_LAYER)
        self.fc  = nn.Linear(D_MODEL, V)

    def forward(self, x):
        x = self.emb(x) + self.pos[:x.size(1)]
        return self.fc(self.tr(x))

model, opt = GPT(), torch.optim.AdamW(GPT().parameters(), lr=LR)
loss_f     = nn.CrossEntropyLoss(ignore_index=PAD_ID)
model, opt, loader = acc.prepare(model, opt, loader)

# ---------- train -----------------------------------------------------------
step = 0
for ep in range(EPOCHS):
    pbar = tqdm(loader, disable=not acc.is_main_process,
                desc=f"ep{ep+1}/{EPOCHS}")
    for x,y in pbar:
        with acc.accumulate(model):
            loss = loss_f(model(x).view(-1, V), y.view(-1))
            acc.backward(loss);  opt.step();  opt.zero_grad()
        step += 1
        if acc.is_main_process:
            pbar.set_postfix(loss=float(loss))
            if step % SAVE_EVERY == 0:
                torch.save({"model": acc.get_state_dict(model),
                            "vocab": tok2id},
                           f"{OUT_DIR}/latest.pt")
                print("💾  latest.pt saved (step", step, ")", flush=True)

if acc.is_main_process:
    torch.save({"model": acc.get_state_dict(model), "vocab": tok2id},
               f"{OUT_DIR}/latest.pt")
    print("✓ finished – latest.pt written")
