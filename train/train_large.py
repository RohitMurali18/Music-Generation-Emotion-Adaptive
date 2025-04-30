# train_tick_10k.py  –  10 000-row run ≤64 GB RAM
# ────────────────────────────────────────────────
CSV_GLOB   = "lmd_full.csv"
MAX_ROWS   = 7500                 # ← only ten-thousand
TICK_MS    = 10

SEQ_LEN    = 512
BATCH_PHYS = 4
ACC_STEPS  = 16
EPOCHS     = 5                      # quick smoke-test
SAVE_EVERY = 5_000

D_MODEL    = 512
N_HEAD     = 8
N_LAYER    = 6
LR         = 3e-4
OUT_DIR    = "ckpt_10k"

# ── std libs ────────────────────────────────────
import os, re, json, pandas as pd, torch, torch.nn as nn, sys
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
to_tick = lambda s: int(round(float(s)*1000/TICK_MS))

def quantise(js: str):
    toks, out = json.loads(js), []
    for t in toks:
        m = note_re.match(t)
        if m:
            p,s,e,d = m.groups()
            out.append(f"[NOTE] [PITCH:{p}] [START_T:{to_tick(s)}] "
                       f"[END_T:{to_tick(e)}] [DUR_T:{to_tick(d)}]")
        else:
            out.append(t)
    return out[:SEQ_LEN]

acc = Accelerator(gradient_accumulation_steps=ACC_STEPS)

# ───────────────── vocab build (rank-0 only) ─────────────────
if acc.is_main_process:
    vocab, rows = {"[PAD]"}, 0
    for p in csv_files:
        for chunk in pd.read_csv(p, usecols=["tokens"], chunksize=2_000):
            for js in chunk["tokens"]:
                vocab.update(quantise(js))
                rows += 1
                if rows % 1_000 == 0:                      # ⇦ live progress
                    print(f"[vocab] {rows:,}/{MAX_ROWS:,}", file=sys.stderr,
                          flush=True)
                if rows >= MAX_ROWS: break
            if rows >= MAX_ROWS: break
        if rows >= MAX_ROWS: break
    tok2id = {t:i for i,t in enumerate(sorted(vocab))}
else:
    tok2id = None                                                 # placeholder

# ───────────────── broadcast to other ranks ──────────────────
if acc.num_processes > 1:
    import torch.distributed as dist
    obj = [tok2id]
    dist.broadcast_object_list(obj, src=0)
    tok2id = obj[0]

PAD_ID     = tok2id["[PAD]"]
VOCAB_SIZE = len(tok2id)
if acc.is_main_process:
    print(f"✓ vocab ready – {VOCAB_SIZE:,} tokens\n", flush=True)

# ───────────────── dataset ───────────────────────────────────
class CSVStream(IterableDataset):
    def __iter__(self):
        seen, bar = 0, None
        if torch.utils.data.get_worker_info() is None:        # main-proc
            bar = tqdm(total=MAX_ROWS, desc="dataset-prep", position=0)
        for p in csv_files:
            for ch in pd.read_csv(p, usecols=["tokens"], chunksize=5_000):
                for js in ch["tokens"]:
                    if seen >= MAX_ROWS:
                        if bar: bar.close()
                        return
                    try:
                        ids = [tok2id[t] for t in quantise(js)]
                    except Exception:
                        continue
                    pad = [PAD_ID]*(SEQ_LEN-len(ids))
                    full = ids + pad
                    if bar:
                        bar.update(1)
                    seen += 1
                    yield (torch.tensor(full[:-1]), torch.tensor(full[1:]))
        if bar:
            bar.close()

dataset = CSVStream()
pin = torch.cuda.is_available()          # avoid warning on CPU
loader = DataLoader(dataset, BATCH_PHYS,
                    num_workers=4, pin_memory=pin)

# ───────────────── model ─────────────────────────────────────
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

# ───────────────── training ─────────────────────────────────
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
                torch.save({"model":acc.get_state_dict(model), "vocab":tok2id},
                           f"{OUT_DIR}/latest.pt")
if acc.is_main_process:
    torch.save({"model":acc.get_state_dict(model), "vocab":tok2id},
               f"{OUT_DIR}/latest.pt")
    print("✓ finished – latest.pt written")
