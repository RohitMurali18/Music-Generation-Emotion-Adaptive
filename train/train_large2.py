# train_large_compact.py
# ----------------------
import glob, json, time, math, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─── config ────────────────────────────────────────────────────────────────
CSV_GLOB          = "lmd_full.csv"     # glob pattern
MAX_ROWS          = 10000            # None → use all rows
SEQ_LEN           = 512                # tokens / sample
D_MODEL           = 512
N_HEAD, N_LAYER   = 8, 6
BATCH             = 16                 # adjust to GPU RAM
EPOCHS            = 6
LR                = 3e-4
SAVE_EVERY_HOURS  = 2                  # autosave cadence
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# tick resolution & caps
RES_MS   = 50                          # 1 tick = 50 ms
MAX_TICK = 4095                        # 0-4095 (≈ 3 min range)
# ───────────────────────────────────────────────────────────────────────────

# ─── helper tables ─────────────────────────────────────────────────────────
PITCHES = [  # C0(0) … B8(107)
    f"{name}{octv}"
    for octv in range(0, 9)
    for name in ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]
]
assert len(PITCHES) == 108

TIME_TOKENS = [f"T_{i}" for i in range(MAX_TICK + 1)]      # 4096
DUR_TOKENS  = [f"DUR_{i}" for i in range(MAX_TICK + 1)]     # 4096

SPECIAL = ["[PAD]", "[START_SEQ]", "[END_SEQ]", "[NOTE]"]

VOCAB = SPECIAL + PITCHES + TIME_TOKENS + DUR_TOKENS
tok2id = {t: i for i, t in enumerate(VOCAB)}
PAD_ID = tok2id["[PAD]"]

# ─── tokenisation ──────────────────────────────────────────────────────────
def ms_to_bucket(ms: float) -> str:
    """quantise millisecond value into 50 ms buckets clamped to MAX_TICK"""
    idx = min(MAX_TICK, int(round(ms / RES_MS)))
    return idx

def explode(js: str):
    """
    Turn the original long token string into a compact, bucketised sequence:
      [START_SEQ] NOTE PITCH  T_x  DUR_y NOTE ...
    """
    seq = [tok2id["[START_SEQ]"]]
    for tok in json.loads(js):
        if not tok.startswith("[NOTE]"):            # skip BPM / KEY etc.
            continue
        # fast parse numbers
        parts = tok.split()
        pitch = parts[1].split(":")[1][:-1]         # 'F#4'
        start = float(parts[2].split(":")[1][:-1])  # seconds
        end   = float(parts[3].split(":")[1][:-1])
        dur   = float(parts[4].split(":")[1][:-2])

        seq.append(tok2id["[NOTE]"])
        seq.append(tok2id[pitch])
        seq.append(tok2id[f"T_{ms_to_bucket(start*1000)}"])
        seq.append(tok2id[f"DUR_{ms_to_bucket(dur*1000)}"])
    seq.append(tok2id["[END_SEQ]"])
    return seq[:SEQ_LEN]

# ─── load CSVs & build dataset ─────────────────────────────────────────────
dfs = [pd.read_csv(p, nrows=MAX_ROWS) for p in glob.glob(CSV_GLOB)]
df  = pd.concat(dfs, ignore_index=True)

class MidiDS(Dataset):
    def __init__(self, series):
        self.data = [explode(js) for js in tqdm(series, desc="tokenising")]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        pad = [PAD_ID] * (SEQ_LEN - len(s))
        s = (s + pad)[:SEQ_LEN]
        return torch.tensor(s[:-1]), torch.tensor(s[1:])

dl = DataLoader(MidiDS(df["tokens"]),
                batch_size=BATCH,
                shuffle=True,
                num_workers=0,
                pin_memory=True)

# ─── model ─────────────────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL))
        block    = nn.TransformerEncoderLayer(
                        D_MODEL, N_HEAD, D_MODEL*4, batch_first=True)
        self.tr = nn.TransformerEncoder(block, N_LAYER)
        self.fc = nn.Linear(D_MODEL, vocab_size)
    def forward(self, x):
        x = self.emb(x) + self.pos[:x.size(1)]
        return self.fc(self.tr(x))

model = GPT(len(VOCAB)).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss(ignore_index=PAD_ID)

def save(tag):
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": tok2id,
            "config": dict(seq_len=SEQ_LEN, d_model=D_MODEL,
                           n_head=N_HEAD, n_layer=N_LAYER,
                           res_ms=RES_MS, max_tick=MAX_TICK)
        },
        f"ckpt_{tag}.pt"
    )

# ─── training loop with timed checkpoints ─────────────────────────────────
last_save = time.time()
interval  = SAVE_EVERY_HOURS * 3600

for ep in range(1, EPOCHS + 1):
    pbar = tqdm(dl, desc=f"epoch {ep}/{EPOCHS}")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = lossf(model(x).view(-1, len(VOCAB)), y.view(-1))
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=float(loss))

        if time.time() - last_save >= interval:
            save(f"ep{ep}_t{int(time.time())}")
            last_save = time.time()

    save(f"ep{ep}")                         # end-of-epoch snapshot

save("final")
print("✓ training complete")
