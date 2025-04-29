# train_tick_cpu.py  ─  fits in ≤64 GB RAM on one m510 node
# ──────────────────────────────────────────────────────────
# * Streams at most 1 000 000 lines from your CSVs
# * 6-layer 512-d transformer, tokens quantised to 10-ms ticks
# * One checkpoint file: ckpt_5h/latest.pt  (over-written every SAVE_EVERY)

# ───────────── user-tunable constants ────────────────────
CSV_GLOB   = "lmd_full.csv"       # change to your dataset location
MAX_ROWS   = 1_000_000                 # stop streaming after this many lines
TICK_MS    = 10                        # 1 tick = 10 ms   (100 Hz grid)

SEQ_LEN    = 512
BATCH_PHYS = 4
ACC_STEPS  = 16                        # logical = 64
EPOCHS     = 20                        # 20 × ~15 min ≈ 5 h
SAVE_EVERY = 10_000                    # overwrite every 10 k updates

D_MODEL    = 512
N_HEAD     = 8
N_LAYER    = 6
LR         = 3e-4
OUT_DIR    = "ckpt_5h"
# ─────────────────────────────────────────────────────────

import os, re, json, pandas as pd, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator
from pathlib import Path
from tqdm.auto import tqdm
from math import ceil

# 0 ─ housekeeping -----------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
Path(OUT_DIR).mkdir(exist_ok=True)

csv_files = list(Path().glob(CSV_GLOB))
assert csv_files, f"No CSV files matched {CSV_GLOB!r}"

note_re = re.compile(
    r"\[NOTE\] \[PITCH:(.+?)\] "
    r"\[START:(.+?)\] \[END:(.+?)\] \[DURATION:(.+?)\]"
)

def to_tick(t: float) -> int:                       # seconds → tick integer
    return int(round((t * 1000) / TICK_MS))

def quantise(seq_str: str):
    """take one JSON-encoded token list, return list[str] with tick tokens"""
    tokens = json.loads(seq_str)
    out = []
    for tok in tokens:
        m = note_re.match(tok)
        if m:
            pitch, start, end, dur = m.groups()
            out.append(f"[NOTE] [PITCH:{pitch}] "
                       f"[START_T:{to_tick(float(start))}] "
                       f"[END_T:{to_tick(float(end))}] "
                       f"[DUR_T:{to_tick(float(dur))}]")
        else:
            out.append(tok)            # [BPM], [KEY_SIGNATURE], [INSTRUMENT], …
    return out[:SEQ_LEN]               # cut long sequences early

# 1 ─ build vocab once (rank 0) ---------------------------------------------
acc = Accelerator(gradient_accumulation_steps=ACC_STEPS, mixed_precision="no")

if acc.is_main_process:
    vocab = {"[PAD]"}
    rows = 0
    for p in csv_files:
        for chunk in pd.read_csv(p, usecols=["tokens"], chunksize=50_000):
            for js in chunk["tokens"]:
                vocab.update(quantise(js))
                rows += 1
                if rows >= MAX_ROWS:
                    break
            if rows >= MAX_ROWS: break
        if rows >= MAX_ROWS: break
    tok2id = {t: i for i, t in enumerate(sorted(vocab))}
else:
    tok2id = None

# broadcast to every rank
tok2id = acc.broadcast_object_list([tok2id], src=0)[0]
PAD_ID = tok2id["[PAD]"]
vocab_size = len(tok2id)
if acc.is_main_process:
    print(f"✓ vocab size {vocab_size:,}  (after tick quantisation)")

# 2 ─ streaming dataset ------------------------------------------------------
class CSVStream(IterableDataset):
    def __iter__(self):
        seen = 0
        for p in csv_files:
            for chunk in pd.read_csv(p, usecols=["tokens"], chunksize=10_000):
                for js in chunk["tokens"]:
                    if seen >= MAX_ROWS:
                        return
                    try:
                        ids = [tok2id[t] for t in quantise(js)]
                    except Exception:
                        continue
                    pad = [PAD_ID]*(SEQ_LEN - len(ids))
                    full = ids + pad
                    yield (torch.tensor(full[:-1], dtype=torch.long),
                           torch.tensor(full[1:],  dtype=torch.long))
                    seen += 1

dataset = CSVStream()
loader  = DataLoader(dataset,
                     batch_size=BATCH_PHYS,
                     num_workers=4,
                     pin_memory=True)

# 3 ─ model ------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL, dtype=torch.float16)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL, dtype=torch.float16))
        blk = nn.TransformerEncoderLayer(
                  D_MODEL, N_HEAD, D_MODEL*4,
                  batch_first=True, dtype=torch.float16)
        self.tr = nn.TransformerEncoder(blk, N_LAYER)
        self.fc = nn.Linear(D_MODEL, vocab, dtype=torch.float16)
    def forward(self, x):
        x = self.emb(x) + self.pos[:x.size(1)]
        return self.fc(self.tr(x))

model  = GPT(vocab_size)
optim  = torch.optim.AdamW(model.parameters(), lr=LR)
loss_f = nn.CrossEntropyLoss(ignore_index=PAD_ID)

model, optim, loader = acc.prepare(model, optim, loader)

# 4 ─ training ---------------------------------------------------------------
step = 0
for epoch in range(EPOCHS):
    pbar = tqdm(loader, disable=not acc.is_main_process,
                desc=f"ep{epoch+1:02d}/{EPOCHS}")
    for x, y in pbar:
        with acc.accumulate(model):
            logits = model(x)
            loss = loss_f(logits.view(-1, vocab_size), y.view(-1))
            acc.backward(loss)
            optim.step(); optim.zero_grad()
        step += 1
        if acc.is_main_process:
            pbar.set_postfix(loss=float(loss))
            if step % SAVE_EVERY == 0:
                torch.save({"model": acc.get_state_dict(model),
                            "vocab": tok2id},
                           f"{OUT_DIR}/latest.pt")
# final save
if acc.is_main_process:
    torch.save({"model": acc.get_state_dict(model), "vocab": tok2id},
               f"{OUT_DIR}/latest.pt")
    print("✓ finished – latest.pt written")
