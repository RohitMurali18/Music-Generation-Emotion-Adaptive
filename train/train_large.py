# train_large.py  – 10 000-row CPU smoke-test with in-memory cache
# --------------------------------------------------------------
CSV_GLOB   = "lmd_full.csv"     # change if your file is elsewhere
MAX_ROWS   = 10_000
SEQ_LEN    = 256                # final sequence length (incl. PADs)
BATCH_PHYS = 4
ACC_STEPS  = 16                 # logical batch = 64
EPOCHS     = 5
SAVE_EVERY = 500              # write latest.pt every N updates
# model
D_MODEL    = 512;  N_HEAD = 8;  N_LAYER = 6;  LR = 3e-4
OUT_DIR    = "ckpt_10k"

# ───────── imports / boiler-plate ───────────────────────────────────────────
import os, re, json, sys, random, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from pathlib import Path
from tqdm.auto import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
Path(OUT_DIR).mkdir(exist_ok=True)

csv_files = list(Path().glob(CSV_GLOB))
assert csv_files, f"No CSV files matched {CSV_GLOB!r}"

note_pat = re.compile(
    r"\[NOTE\] \[PITCH:(.+?)\] "
    r"\[START:(.+?)\] \[END:(.+?)\] \[DURATION:(.+?)\]"
)
TICK_MS = 10
to_tick = lambda s: int(round(float(s) * 1000 / TICK_MS))

def explode(js: str):
    """Convert seconds→ticks and split NOTE line into atomic subtokens."""
    out = []
    for tok in json.loads(js):
        m = note_pat.match(tok)
        if m is None:           # not a NOTE line
            out.append(tok)
            continue
        p, s, e, d = m.groups()
        out.extend(("[NOTE]", "[PITCH]", p,
                    "[START_T]", str(to_tick(s)),
                    "[END_T]",   str(to_tick(e)),
                    "[DUR_T]",   str(to_tick(d))))
    return out[:SEQ_LEN]        # hard truncate

# ───────── Accelerator setup ────────────────────────────────────────────────
acc = Accelerator(gradient_accumulation_steps=ACC_STEPS)

# ───────── pass 1: read ≤MAX_ROWS, build vocab, keep token strings ----------

token_seqs, vocab = [], {"[PAD]"}
rows = 0
for path in csv_files:
    for chunk in pd.read_csv(path, usecols=["tokens"], chunksize=2_000):
        for js in chunk["tokens"]:
            toks = explode(js)
            token_seqs.append(toks)
            vocab.update(toks)
            rows += 1
            if rows % 1_000 == 0 and acc.is_main_process:
                print(f"[vocab] {rows:,}/{MAX_ROWS:,}", flush=True)
            if rows >= MAX_ROWS:
                break
        if rows >= MAX_ROWS: break
    if rows >= MAX_ROWS: break

tok2id = {t: i for i, t in enumerate(sorted(vocab))}
PAD_ID  = tok2id["[PAD]"]
VOCAB   = len(tok2id)
if acc.is_main_process:
    print(f"✓ vocab ready – {VOCAB:,} tokens\n", flush=True)

# ───────── pass 2: convert strings → id tensors (cached in RAM) -------------
cached = []
for toks in tqdm(token_seqs, desc="tensor-cache", disable=not acc.is_main_process):
    ids = [tok2id[t] for t in toks]
    if len(ids) < SEQ_LEN:
        ids.extend([PAD_ID] * (SEQ_LEN - len(ids)))
    else:
        ids = ids[:SEQ_LEN]
    cached.append(torch.tensor(ids, dtype=torch.long))  # shape [256]

del token_seqs, vocab     # free string memory ASAP

# ───────── dataset backed by cached tensors ─────────────────────────────────
class CachedDS(Dataset):
    def __len__(self): return len(cached)
    def __getitem__(self, idx):
        seq = cached[idx]
        return seq[:-1], seq[1:]                       # both length 255

dataset = CachedDS()
loader  = DataLoader(dataset,
                     batch_size=BATCH_PHYS,
                     shuffle=True,      # shuffle indices each epoch
                     num_workers=0,
                     pin_memory=False)

# ───────── tiny GPT-like model ──────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL))
        enc = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL*4,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(enc, N_LAYER)
        self.fc = nn.Linear(D_MODEL, V)
    def forward(self, x):
        return self.fc(self.tr(self.emb(x) + self.pos[:x.size(1)]))

model  = GPT(VOCAB)
optim  = torch.optim.AdamW(model.parameters(), lr=LR)
loss_f = nn.CrossEntropyLoss(ignore_index=PAD_ID)

model, optim, loader = acc.prepare(model, optim, loader)

# ───────── training loop ────────────────────────────────────────────────────
step = 0
for ep in range(EPOCHS):
    pbar = tqdm(loader, disable=not acc.is_main_process,
                desc=f"ep{ep+1}/{EPOCHS}")
    for x, y in pbar:
        with acc.accumulate(model):
            loss = loss_f(model(x).view(-1, VOCAB), y.view(-1))
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
