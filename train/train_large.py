# train_tick_cpu.py  –  ≤64 GB RAM  (patched broadcast)
# ──────────────────────────────────────────────────────
CSV_GLOB   = "lmd_full.csv"
MAX_ROWS   = 1_000_000
TICK_MS    = 10

SEQ_LEN    = 512
BATCH_PHYS = 4
ACC_STEPS  = 16
EPOCHS     = 20
SAVE_EVERY = 10_000

D_MODEL    = 512
N_HEAD     = 8
N_LAYER    = 6
LR         = 3e-4
OUT_DIR    = "ckpt_5h"

import os, re, json, pandas as pd, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator
from pathlib import Path
from tqdm.auto import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
Path(OUT_DIR).mkdir(exist_ok=True)

csv_files = list(Path().glob(CSV_GLOB))
assert csv_files, f"No CSV files matched {CSV_GLOB!r}"

note_re = re.compile(
    r"\[NOTE\] \[PITCH:(.+?)\] "
    r"\[START:(.+?)\] \[END:(.+?)\] \[DURATION:(.+?)\]"
)
def to_tick(sec): return int(round(sec*1000/TICK_MS))
def quantise(js):
    toks, out = json.loads(js), []
    for t in toks:
        m = note_re.match(t)
        if m:
            p, s, e, d = m.groups()
            out.append(f"[NOTE] [PITCH:{p}] [START_T:{to_tick(float(s))}] "
                       f"[END_T:{to_tick(float(e))}] [DUR_T:{to_tick(float(d))}]")
        else:
            out.append(t)
    return out[:SEQ_LEN]

acc = Accelerator(gradient_accumulation_steps=ACC_STEPS)

# ── vocab (only rank-0 reads files) ───────────────────
if acc.is_main_process:
    vocab, rows = {"[PAD]"}, 0
    for p in csv_files:
        for ch in pd.read_csv(p, usecols=["tokens"], chunksize=50_000):
            for js in ch["tokens"]:
                vocab.update(quantise(js)); rows += 1
                if rows >= MAX_ROWS: break
            if rows >= MAX_ROWS: break
        if rows >= MAX_ROWS: break
    tok2id = {t:i for i,t in enumerate(sorted(vocab))}
else:
    tok2id = None

# ── broadcast to the other ranks (patched) ────────────
if acc.num_processes > 1:
    import torch.distributed as dist
    obj = [tok2id]
    dist.broadcast_object_list(obj, src=0)
    tok2id = obj[0]

PAD_ID     = tok2id["[PAD]"]
VOCAB_SIZE = len(tok2id)
if acc.is_main_process:
    print(f"✓ vocab size {VOCAB_SIZE:,}")

# ── streaming dataset ────────────────────────────────
class CSVStream(IterableDataset):
    def __iter__(self):
        seen = 0
        for p in csv_files:
            for ch in pd.read_csv(p, usecols=["tokens"], chunksize=10_000):
                for js in ch["tokens"]:
                    if seen >= MAX_ROWS: return
                    try:
                        ids = [tok2id[t] for t in quantise(js)]
                    except Exception:
                        continue
                    pad = [PAD_ID]*(SEQ_LEN-len(ids)); full = ids+pad
                    yield (torch.tensor(full[:-1]), torch.tensor(full[1:]))
                    seen += 1
dataset = CSVStream()
loader  = DataLoader(dataset, BATCH_PHYS, num_workers=4, pin_memory=True)

# ── model ─────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(s, V):
        super().__init__()
        s.emb = nn.Embedding(V, D_MODEL, dtype=torch.float16)
        s.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL, dtype=torch.float16))
        blk   = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL*4,
                                           batch_first=True, dtype=torch.float16)
        s.tr  = nn.TransformerEncoder(blk, N_LAYER)
        s.fc  = nn.Linear(D_MODEL, V, dtype=torch.float16)
    def forward(s,x): return s.fc(s.tr(s.emb(x)+s.pos[:x.size(1)]))

model  = GPT(VOCAB_SIZE)
optim  = torch.optim.AdamW(model.parameters(), lr=LR)
loss_f = nn.CrossEntropyLoss(ignore_index=PAD_ID)

model, optim, loader = acc.prepare(model, optim, loader)

# ── training loop ────────────────────────────────────
step = 0
for ep in range(EPOCHS):
    pbar = tqdm(loader, disable=not acc.is_main_process,
                desc=f"ep{ep+1}/{EPOCHS}")
    for x,y in pbar:
        with acc.accumulate(model):
            loss = loss_f(model(x).view(-1, VOCAB_SIZE), y.view(-1))
            acc.backward(loss); optim.step(); optim.zero_grad()
        step += 1
        if acc.is_main_process:
            pbar.set_postfix(loss=float(loss))
            if step % SAVE_EVERY == 0:
                torch.save({"model":acc.get_state_dict(model),"vocab":tok2id},
                           f"{OUT_DIR}/latest.pt")
if acc.is_main_process:
    torch.save({"model":acc.get_state_dict(model),"vocab":tok2id},
               f"{OUT_DIR}/latest.pt")
    print("✓ finished – latest.pt written")
