"""
train_cpu_5h.py  –  single-m510 run that ends ~5 h later
─────────────────────────────────────────────────────────
• Reads at most 1 000 000 rows across all CSVs (streaming, so low RAM)
• 6-layer × 512-d causal Transformer
"""

# ───────────── tweak here only ────────────────────────
CSV_GLOB   = "lmd_mini.csv"    # adjust if your files live elsewhere
MAX_ROWS   = 1_000_000              # hard cap: rows streamed then stop
SEQ_LEN    = 512
BATCH_PHYS = 4                      # per process
ACC_STEPS  = 16                     # logical batch 64
EPOCHS     = 20                     # 20 × ~15 min ≈ 5 h
LR         = 3e-4

D_MODEL    = 512
N_HEAD     = 8
N_LAYER    = 6
SAVE_EVERY = 10_000                 # save every 10 k updates
OUT_DIR    = "ckpt_5h"
# ──────────────────────────────────────────────────────

# ---------- boilerplate (same as previous large script, trimmed) ------------
import os, json, pandas as pd, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator
from pathlib import Path
from tqdm.auto import tqdm

# limit math threads so 8 processes don’t oversubscribe
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

csv_files = list(Path().glob(CSV_GLOB))
assert csv_files, "No CSV files found – check CSV_GLOB"

# 1) build vocab (stream first CSV) ------------------------------------------
vocab = {"[PAD]"}
rows_seen = 0
for path in csv_files:
    for chunk in pd.read_csv(path, usecols=["tokens"], chunksize=50_000):
        for js in chunk["tokens"]:
            vocab.update(json.loads(js))
            rows_seen += 1
            if rows_seen >= MAX_ROWS:
                break
        if rows_seen >= MAX_ROWS:
            break
    if rows_seen >= MAX_ROWS:
        break

tok2id = {t: i for i, t in enumerate(sorted(vocab))}
PAD_ID = tok2id["[PAD]"]
print(f"✓ vocab size {len(tok2id):,} from {rows_seen:,} sampled rows")

# 2) streaming dataset -------------------------------------------------------
class CSVStream(IterableDataset):
    def __iter__(self):
        seen = 0
        for p in csv_files:
            for chunk in pd.read_csv(p, usecols=["tokens"], chunksize=10_000):
                for js in chunk["tokens"]:
                    if seen >= MAX_ROWS: return
                    try:
                        ids = [tok2id[t] for t in json.loads(js)][:SEQ_LEN]
                    except Exception:
                        continue
                    pad = [PAD_ID]*(SEQ_LEN-len(ids))
                    full= ids+pad
                    yield (torch.tensor(full[:-1]),
                           torch.tensor(full[1:]))
                    seen += 1

dataset = CSVStream()

# 3) model -------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL))
        blk = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL*4,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(blk, N_LAYER)
        self.fc = nn.Linear(D_MODEL, vocab)
    def forward(self, x):
        return self.fc(self.tr(self.emb(x) + self.pos[:x.size(1)]))

model = GPT(len(tok2id))
optim = torch.optim.AdamW(model.parameters(), lr=LR)
loss_f = nn.CrossEntropyLoss(ignore_index=PAD_ID)

# 4) accelerator -------------------------------------------------------------
acc = Accelerator(gradient_accumulation_steps=ACC_STEPS, mixed_precision="no")
loader = DataLoader(dataset, batch_size=BATCH_PHYS, num_workers=4,
                    pin_memory=True)
model, optim, loader = acc.prepare(model, optim, loader)

Path(OUT_DIR).mkdir(exist_ok=True)

LATEST = None
# 5) train -------------------------------------------------------------------
for epoch in range(EPOCHS):
    pbar = tqdm(loader, disable=not acc.is_main_process,
                desc=f"ep{epoch+1}/{EPOCHS}")
    for x, y in pbar:
        with acc.accumulate(model):
            logits = model(x)
            loss = loss_f(logits.view(-1, logits.size(-1)), y.view(-1))
            acc.backward(loss)
            optim.step(); optim.zero_grad()
        step += 1

        if acc.is_main_process:
            pbar.set_postfix(loss=float(loss))

            # ────── save only the newest ──────
            if step % SAVE_EVERY == 0:
                new_ckpt = Path(OUT_DIR) / f"step{step:07d}.pt"
                torch.save({"model": acc.get_state_dict(model),
                            "vocab": tok2id}, new_ckpt)

                # delete the previous file (if any)
                if LATEST and LATEST.exists():
                    LATEST.unlink(missing_ok=True)
                LATEST = new_ckpt
                print("💾  saved step", step)
# final
if acc.is_main_process:
    torch.save({"model": acc.get_state_dict(model), "vocab": tok2id},
               f"{OUT_DIR}/final.pt")
    print("✓ done – final checkpoint written")
