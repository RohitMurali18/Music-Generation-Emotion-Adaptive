# train_large.py  –  10 000-row CPU smoke test (≤ 64 GB RAM)
# ───────────────────────────────────────────────────────────
CSV_GLOB   = "lmd_full.csv"
MAX_ROWS   = 5000
SEQ_LEN    = 256                     # final length of each sequence
BATCH_PHYS = 8
ACC_STEPS  = 8                      # logical batch 64
EPOCHS     = 6
SAVE_EVERY = 500

D_MODEL    = 256
N_HEAD     = 8
N_LAYER    = 4
LR         = 3e-4
OUT_DIR    = "ckpt_10k4"

# ─── std-lib / deps ─────────────────────────────────────────────────────────
import os, re, json, sys, pandas as pd, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator
from pathlib import Path
from tqdm.auto import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
Path(OUT_DIR).mkdir(exist_ok=True)

csv_files = list(Path().glob(CSV_GLOB))
assert csv_files, f"No CSV files matched {CSV_GLOB!r}"

# ─── token helpers ──────────────────────────────────────────────────────────
note_pat_secs = re.compile(
    r"\[NOTE\] \[PITCH:(.+?)\] "
    r"\[START:(.+?)\] \[END:(.+?)\] \[DURATION:(.+?)\]"
)
TICK_MS = 10
to_tick = lambda s: int(round(float(s) * 1000 / TICK_MS))

def explode(js: str):
    """seconds → ticks, split NOTE line into atomic subtokens"""
    out = []
    for tok in json.loads(js):
        m = note_pat_secs.match(tok)
        if not m:
            out.append(tok)
            continue
        p, s, e, d = m.groups()
        out.extend((
            "[NOTE]", "[PITCH]", p,
            "[START_T]", str(to_tick(s)),
            "[END_T]",   str(to_tick(e)),
            "[DUR_T]",   str(to_tick(d)),
        ))
    # *** clamp length here ***
    return out[:SEQ_LEN]

# ─── Accelerator init -------------------------------------------------------
acc = Accelerator(gradient_accumulation_steps=ACC_STEPS)

# ─── vocab build (rank-0 only) ---------------------------------------------
if acc.is_main_process:
    vocab, rows = {"[PAD]"}, 0
    for p in csv_files:
        for chunk in pd.read_csv(p, usecols=["tokens"], chunksize=2_000):
            for js in chunk["tokens"]:
                vocab.update(explode(js))
                rows += 1
                if rows % 1_000 == 0:
                    print(f"[vocab] {rows:,}/{MAX_ROWS:,}", file=sys.stderr,
                          flush=True)
                if rows >= MAX_ROWS:
                    break
            if rows >= MAX_ROWS:
                break
        if rows >= MAX_ROWS:
            break
    tok2id = {t: i for i, t in enumerate(sorted(vocab))}
else:
    tok2id = None                                    # placeholder

# ─── broadcast small Python object (if >1 rank) -----------------------------
if acc.num_processes > 1:
    import torch.distributed as dist
    obj = [tok2id]
    dist.broadcast_object_list(obj, src=0)
    tok2id = obj[0]

PAD_ID = tok2id["[PAD]"]
VOCAB  = len(tok2id)
if acc.is_main_process:
    print(f"✓ vocab ready – {VOCAB:,} tokens\n", flush=True)

# ─── dataset / loader -------------------------------------------------------
class CSVStream(IterableDataset):
    def __iter__(self):
        seen = 0
        bar = tqdm(total=MAX_ROWS, desc="dataset-prep", position=0) \
              if acc.is_main_process else None
        for p in csv_files:
            for chunk in pd.read_csv(p, usecols=["tokens"], chunksize=5_000):
                for js in chunk["tokens"]:
                    if seen >= MAX_ROWS:
                        if bar: bar.close()
                        return
                    ids = [tok2id[t] for t in explode(js)]
                    if len(ids) < SEQ_LEN:                      # pad up
                        ids.extend([PAD_ID] * (SEQ_LEN - len(ids)))
                    else:                                      # clamp down
                        ids = ids[:SEQ_LEN]
                    full = ids                                  # exact 256
                    seen += 1
                    if bar: bar.update(1)
                    yield (torch.tensor(full[:-1]),
                           torch.tensor(full[1:]))
        if bar:
            bar.close()

dataset = CSVStream()
loader  = DataLoader(dataset,
                     batch_size=BATCH_PHYS,
                     num_workers=0,       # keep 0 → easiest debug
                     pin_memory=False)

# ─── model ------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN - 1, D_MODEL))
        blk = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL * 4,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(blk, N_LAYER)
        self.fc = nn.Linear(D_MODEL, V)

    def forward(self, x):
        return self.fc(self.tr(self.emb(x) + self.pos[:x.size(1)]))

model  = GPT(VOCAB)
optim  = torch.optim.AdamW(model.parameters(), lr=LR)
loss_f = nn.CrossEntropyLoss(ignore_index=PAD_ID)

model, optim, loader = acc.prepare(model, optim, loader)

# ─── training ---------------------------------------------------------------
step = 0
for ep in range(EPOCHS):
    pbar = tqdm(loader, disable=not acc.is_main_process,
                desc=f"ep{ep+1}/{EPOCHS}")
    for x, y in pbar:
        with acc.accumulate(model):
            logits = model(x)
            loss   = loss_f(logits.view(-1, VOCAB), y.view(-1))
            acc.backward(loss)
            optim.step(); optim.zero_grad()
        step += 1
        if acc.is_main_process:
            pbar.set_postfix(loss=float(loss))
            if step % SAVE_EVERY == 0:
                torch.save({"model": acc.get_state_dict(model),
                            "vocab": tok2id},
                           f"{OUT_DIR}/latest.pt")
if acc.is_main_process:
    torch.save({"model": acc.get_state_dict(model), "vocab": tok2id},
               f"{OUT_DIR}/latest.pt")
    print("✓ finished – latest.pt written")