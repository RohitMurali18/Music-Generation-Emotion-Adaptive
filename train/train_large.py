#!/usr/bin/env python3
# train_large.py  –  10 000-row smoke test, ≤64 GB RAM
# ═════════════════════════════════════════════════════

# ─────────── configuration ──────────────────────────────────────────
CSV_GLOB   = "lmd_full.csv"     # glob or "**/*.csv" is fine
MAX_ROWS   = 7500            # total samples streamed (stop afterwards)
SEQ_LEN    = 512
BATCH_PHYS = 4
ACC_STEPS  = 16                 # logical batch = 64
EPOCHS     = 5
SAVE_EVERY = 5_000              # optimisation steps (not epochs)

D_MODEL    = 512
N_HEAD     = 8
N_LAYER    = 6
LR         = 3e-4
OUT_DIR    = "ckpt_10k"
TICK_MS    = 10                 # note-time quantisation (optional)

# ─────────── std-lib & deps ─────────────────────────────────────────
import os, sys, json, re, glob, itertools
from pathlib import Path

import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

# ─────────── housekeeping ─────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"     # don’t oversubscribe
torch.set_num_threads(1)
Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

csv_files = sorted(glob.glob(CSV_GLOB, recursive=True))
assert csv_files, f"No CSV files matched {CSV_GLOB!r}"

# ─────────── helper: quantise NOTE tokens ──────────────────────────
note_re = re.compile(
    r"\[NOTE\] \[PITCH:(.+?)\] "
    r"\[START:(.+?)\] \[END:(.+?)\] \[DURATION:(.+?)\]"
)
to_tick = lambda s: int(round(float(s) * 1000 / TICK_MS))

def quantise(js: str):
    toks, out = json.loads(js), []
    for t in toks:
        m = note_re.match(t)
        if m:                                  # replace float-seconds → ticks
            p, s, e, d = m.groups()
            out.append(
                f"[NOTE] [PITCH:{p}] [START_T:{to_tick(s)}] "
                f"[END_T:{to_tick(e)}] [DUR_T:{to_tick(d)}]"
            )
        else:
            out.append(t)
    return out[:SEQ_LEN]

# ─────────── Accelerator init (handles 1 CPU process) ──────────────
acc = Accelerator(gradient_accumulation_steps=ACC_STEPS)

# ─────────── vocab build  (rank-0 only) ────────────────────────────
if acc.is_main_process:
    vocab, rows = {"[PAD]"}, 0
    for p in csv_files:
        for chunk in pd.read_csv(p, usecols=["tokens"], chunksize=1_000,
                                 engine="c", dtype={"tokens": "string"}):
            for js in chunk["tokens"]:
                vocab.update(quantise(js))
                rows += 1
                if rows % 1_000 == 0 or rows == MAX_ROWS:
                    print(f"[vocab] {rows:,}/{MAX_ROWS:,}", flush=True)
                if rows >= MAX_ROWS:
                    break
            if rows >= MAX_ROWS:
                break
        if rows >= MAX_ROWS:
            break
    tok2id = {t: i for i, t in enumerate(sorted(vocab))}
else:
    tok2id = None                                   # placeholder

# ─────────── share vocab if >1 ranks (here we have only 1) ─────────
if acc.num_processes > 1:
    import torch.distributed as dist
    obj = [tok2id]
    dist.broadcast_object_list(obj, src=0)
    tok2id = obj[0]

PAD_ID      = tok2id["[PAD]"]
VOCAB_SIZE  = len(tok2id)
if acc.is_main_process:
    print(f"✓ vocab ready – {VOCAB_SIZE:,} tokens\n", flush=True)

# ─────────── streaming dataset with heartbeat ──────────────────────
class CSVStream(IterableDataset):
    def __iter__(self):
        seen = 0
        for path in csv_files:
            for chunk in pd.read_csv(
                    path,
                    usecols=["tokens"],
                    chunksize=1_000,
                    engine="c",
                    dtype={"tokens": "string"}):
                for js in chunk["tokens"]:
                    if seen >= MAX_ROWS:
                        return
                    try:
                        ids = [tok2id[t] for t in itertools.islice(
                               quantise(js), SEQ_LEN)]
                    except Exception:
                        continue
                    full = ids + [PAD_ID] * (SEQ_LEN - len(ids))
                    seen += 1
                    if seen % 1_000 == 0:
                        # heartbeat is **not** swallowed by tqdm
                        print(f"[loader] streamed {seen:,} / {MAX_ROWS:,}",
                              flush=True)
                    yield (torch.tensor(full[:-1]),
                           torch.tensor(full[1:]))

dataset = CSVStream()
loader  = DataLoader(dataset,
                     batch_size=BATCH_PHYS,
                     num_workers=0,      # keep 0 → easier to debug
                     pin_memory=False)

# ─────────── tiny GPT model (float16 to save RAM) ──────────────────
class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL, dtype=torch.float16)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN - 1, D_MODEL,
                                            dtype=torch.float16))
        blk = nn.TransformerEncoderLayer(
            D_MODEL, N_HEAD, D_MODEL * 4,
            batch_first=True, dtype=torch.float16)
        self.tr = nn.TransformerEncoder(blk, N_LAYER)
        self.fc = nn.Linear(D_MODEL, V, dtype=torch.float16)

    def forward(self, x):
        return self.fc(self.tr(self.emb(x) + self.pos[:x.size(1)]))

model  = GPT(VOCAB_SIZE)
optim  = torch.optim.AdamW(model.parameters(), lr=LR)
loss_f = nn.CrossEntropyLoss(ignore_index=PAD_ID)

model, optim, loader = acc.prepare(model, optim, loader)

# ─────────── training loop ─────────────────────────────────────────
step = 0
for ep in range(EPOCHS):
    pbar = tqdm(loader,
                disable=not acc.is_main_process,
                desc=f"ep{ep+1}/{EPOCHS}")
    for x, y in pbar:
        with acc.accumulate(model):
            loss = loss_f(model(x).view(-1, VOCAB_SIZE), y.view(-1))
            acc.backward(loss)
            optim.step()
            optim.zero_grad()
        step += 1
        if acc.is_main_process:
            pbar.set_postfix(loss=float(loss))
            if step % SAVE_EVERY == 0:
                torch.save(
                    {"model": acc.get_state_dict(model), "vocab": tok2id},
                    f"{OUT_DIR}/latest.pt")
                print("💾  latest.pt saved (step", step, ")", flush=True)

# final checkpoint
if acc.is_main_process:
    torch.save({"model": acc.get_state_dict(model), "vocab": tok2id},
               f"{OUT_DIR}/latest.pt")
    print("✓ training finished – latest.pt written")
