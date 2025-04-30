#!/usr/bin/env python3
# train_large.py  – 10 k-row CPU demo, ≤ 64 GB RAM
# ══════════════════════════════════════════════════
CSV_PATH   = "lmd_full.csv"     # one csv or glob
MAX_ROWS   = 10_000
SEQ_LEN    = 256                # we shortened – easier on RAM
BATCH      = 8
ACC_STEPS  = 8                  # logical batch 64
EPOCHS     = 5
SAVE_EVERY = 2_000

D_MODEL    = 256
N_HEAD     = 4
N_LAYER    = 4
LR         = 3e-4
TICK_MS    = 10                 # quantise 1 tick = 10 ms
OUT_DIR    = "ckpt_10k"

# ────────────────────────────────────────────────
import os, sys, json, math, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

os.environ["OMP_NUM_THREADS"] = "1";  torch.set_num_threads(1)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ——— helper ————————————————————————————————————
note_fields = ["[NOTE]", "[PITCH]", "[START_T]", "[END_T]", "[DUR_T]"]
def split_note(tok:str):
    """"[NOTE] [PITCH:A4] …" -> list of 9 atomic tokens"""
    parts = tok.split()
    if parts[0] != "[NOTE]":      # safety
        return [tok]
    # parts = ['[NOTE]', '[PITCH:A4]', '[START_T:982]', '[END_T:996]', '[DUR_T:14]']
    pitch = parts[1][len("[PITCH:"):-1]           # A4
    s  = int(parts[2][len("[START_T:"):-1])
    e  = int(parts[3][len("[END_T:"):-1])
    d  = int(parts[4][len("[DUR_T:"):-1])
    return ["[NOTE]", "[PITCH]", pitch,
            "[START_T]", str(s),
            "[END_T]",   str(e),
            "[DUR_T]",   str(d)]

def explode(js: str):
    out = []
    for t in json.loads(js):
        if t.startswith("[NOTE]"):
            out.extend(split_note(t))
        else:
            out.append(t)
        if len(out) >= SEQ_LEN:
            break
    return out

# ——— accelerator —————————————————————————————————
acc = Accelerator(gradient_accumulation_steps=ACC_STEPS)

# ——— vocab build (rank 0) ——————————————————————
if acc.is_main_process:
    vocab, rows = {"[PAD]"}, 0
    for chunk in pd.read_csv(CSV_PATH, usecols=["tokens"],
                             chunksize=500, dtype={"tokens":"string"}):
        for js in chunk["tokens"]:
            vocab.update(explode(js));  rows += 1
            if rows % 1000 == 0 or rows == MAX_ROWS:
                print(f"[vocab] {rows:,}/{MAX_ROWS:,}", flush=True)
            if rows >= MAX_ROWS:
                break
        if rows >= MAX_ROWS:
            break
    tok2id = {t:i for i,t in enumerate(sorted(vocab))}
else:
    tok2id = None

# broadcast to workers (if any)
if acc.num_processes > 1:
    import torch.distributed as dist
    obj = [tok2id];  dist.broadcast_object_list(obj, 0);  tok2id = obj[0]

PAD_ID,  V = tok2id["[PAD]"], len(tok2id)
if acc.is_main_process:
    print(f"✓ vocab size {V:,}\n", flush=True)

# ——— dataset ————————————————————————————————
class Stream(IterableDataset):
    def __iter__(self):
        seen = 0
        for chunk in pd.read_csv(CSV_PATH, usecols=["tokens"],
                                 chunksize=1000, dtype={"tokens":"string"}):
            for js in chunk["tokens"]:
                if seen >= MAX_ROWS:
                    return
                ids = [tok2id[t] for t in explode(js)]
                full = ids + [PAD_ID]*(SEQ_LEN-len(ids))
                seen += 1
                if seen % 1000 == 0:
                    print(f"[loader] {seen:,}/{MAX_ROWS:,}", flush=True)
                yield (torch.tensor(full[:-1]), torch.tensor(full[1:]))

loader = DataLoader(Stream(), BATCH, num_workers=0, pin_memory=False)

# ——— model ————————————————————————————————
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL))
        blk = nn.TransformerEncoderLayer(
                D_MODEL, N_HEAD, D_MODEL*4, batch_first=True)
        self.tr  = nn.TransformerEncoder(blk, N_LAYER)
        self.fc  = nn.Linear(D_MODEL, V)

    def forward(self, x):
        x = self.emb(x) + self.pos[:x.size(1)]
        return self.fc(self.tr(x))

model, opt = GPT(), torch.optim.AdamW(GPT().parameters(), lr=LR)
loss_f = nn.CrossEntropyLoss(ignore_index=PAD_ID)

model, opt, loader = acc.prepare(model, opt, loader)

# ——— training ——————————————————————————————
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
                torch.save({"model":acc.get_state_dict(model),"vocab":tok2id},
                           f"{OUT_DIR}/latest.pt")
                print("💾 saved latest.pt (step", step, ")", flush=True)

if acc.is_main_process:
    torch.save({"model":acc.get_state_dict(model),"vocab":tok2id},
               f"{OUT_DIR}/latest.pt")
    print("✓ finished – latest.pt written")
