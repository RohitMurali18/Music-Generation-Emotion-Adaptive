# ── config ────────────────────────────────────────────────────────────────
CSV_GLOB   = "lmd_full.csv"     # glob or single path
MAX_ROWS   = 10_000
hparams    = dict(
    seq_len = 512,
    d_model = 512,
    n_head  = 8,
    n_layer = 6,
)
BATCH_PHYS = 8
ACC_STEPS  = 8
EPOCHS     = 6
LR         = 3e-4
OUT_DIR    = "ckpt_full"
SAVE_EVERY = 500
# ──────────────────────────────────────────────────────────────────────────

import os, re, json, sys, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator
from pathlib import Path
from tqdm.auto import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
Path(OUT_DIR).mkdir(exist_ok=True)
csv_files = list(Path().glob(CSV_GLOB))
assert csv_files, f"no CSV matched {CSV_GLOB}"

note_pat = re.compile(r"\[NOTE\] \[PITCH:(.+?)\] \[START:(.+?)\] "
                      r"\[END:(.+?)\] \[DURATION:(.+?)\]")
TICK_MS = 10
to_tick = lambda s: int(round(float(s) * 1000 / TICK_MS))

def explode(js):
    out = []
    for tok in json.loads(js):
        m = note_pat.match(tok)
        if not m:
            out.append(tok); continue
        p, s, e, d = m.groups()
        out += ["[NOTE]", "[PITCH]", p,
                "[START_T]", str(to_tick(s)),
                "[END_T]",   str(to_tick(e)),
                "[DUR_T]",   str(to_tick(d))]
    return out[:hparams["seq_len"]]

acc = Accelerator(gradient_accumulation_steps=ACC_STEPS)

# vocab
if acc.is_main_process:
    vocab, rows = {"[PAD]"}, 0
    for p in csv_files:
        for chunk in pd.read_csv(p, usecols=["tokens"], chunksize=2_000):
            for js in chunk["tokens"]:
                vocab.update(explode(js)); rows += 1
                if rows >= MAX_ROWS: break
            if rows >= MAX_ROWS: break
        if rows >= MAX_ROWS: break
    tok2id = {t: i for i, t in enumerate(sorted(vocab))}
else:
    tok2id = None
if acc.num_processes > 1:
    import torch.distributed as dist
    obj = [tok2id]; dist.broadcast_object_list(obj, src=0); tok2id = obj[0]

PAD_ID, VOCAB = tok2id["[PAD]"], len(tok2id)
if acc.is_main_process: print(f"vocab {VOCAB}")

class CSVStream(IterableDataset):
    def __iter__(self):
        seen = 0
        bar = tqdm(total=MAX_ROWS, disable=not acc.is_main_process)
        for p in csv_files:
            for chunk in pd.read_csv(p, usecols=["tokens"], chunksize=5_000):
                for js in chunk["tokens"]:
                    if seen >= MAX_ROWS: bar.close(); return
                    ids = [tok2id[t] for t in explode(js)]
                    pad = hparams["seq_len"] - len(ids)
                    ids = ids + [PAD_ID]*pad if pad > 0 else ids[:hparams["seq_len"]]
                    seen += 1; bar.update(1)
                    yield (torch.tensor(ids[:-1]), torch.tensor(ids[1:]))
        bar.close()

class GPT(nn.Module):
    def __init__(self, V, hp):
        super().__init__()
        self.emb = nn.Embedding(V, hp["d_model"])
        self.pos = nn.Parameter(torch.zeros(hp["seq_len"]-1, hp["d_model"]))
        blk = nn.TransformerEncoderLayer(hp["d_model"], hp["n_head"],
                                         hp["d_model"]*4, batch_first=True)
        self.tr = nn.TransformerEncoder(blk, hp["n_layer"])
        self.fc = nn.Linear(hp["d_model"], V)
    def forward(self, x):
        return self.fc(self.tr(self.emb(x) + self.pos[:x.size(1)]))

model  = GPT(VOCAB, hparams)
optim  = torch.optim.AdamW(model.parameters(), lr=LR)
loss_f = nn.CrossEntropyLoss(ignore_index=PAD_ID)
loader = DataLoader(CSVStream(), batch_size=BATCH_PHYS, num_workers=0)

model, optim, loader = acc.prepare(model, optim, loader)

step = 0
for ep in range(EPOCHS):
    pbar = tqdm(loader, disable=not acc.is_main_process,
                desc=f"ep{ep+1}/{EPOCHS}")
    for x, y in pbar:
        with acc.accumulate(model):
            loss = loss_f(model(x).view(-1, VOCAB), y.view(-1))
            acc.backward(loss); optim.step(); optim.zero_grad()
        step += 1
        if acc.is_main_process:
            pbar.set_postfix(loss=float(loss))
            if step % SAVE_EVERY == 0:
                torch.save({"model": acc.get_state_dict(model),
                            "vocab": tok2id,
                            "hparams": hparams},
                           f"{OUT_DIR}/latest.pt")
if acc.is_main_process:
    torch.save({"model": acc.get_state_dict(model),
                "vocab": tok2id,
                "hparams": hparams},
               f"{OUT_DIR}/latest.pt")
    print("done, saved latest.pt")
