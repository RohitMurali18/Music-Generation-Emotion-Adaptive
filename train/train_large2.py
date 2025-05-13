# train_large_compact.py  –  memory-friendly music-token GPT
import glob, json, time, re, math, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────
CSV_GLOB         = "lmd_full.csv"   # file or glob
MAX_ROWS         = 10000         # None → all rows
SEQ_LEN          = 512
D_MODEL          = 512
N_HEAD, N_LAYER  = 8, 6
BATCH            = 16
EPOCHS           = 6
LR               = 3e-4
SAVE_HOURS       = 2
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

RES_MS, MAX_TICK = 50, 4095         # 50 ms buckets, 0-4095 range
# ──────────────────────────────────────────────────────────────────────────

# ── vocabulary ────────────────────────────────────────────────────────────
SPECIAL      = ["[PAD]", "[START_SEQ]", "[END_SEQ]", "[NOTE]"]
PITCH_TOKENS = [f"P_{i}"      for i in range(128)]          # MIDI 0-127
TIME_TOKENS  = [f"T_{i}"      for i in range(MAX_TICK + 1)]
DUR_TOKENS   = [f"DUR_{i}"    for i in range(MAX_TICK + 1)]

VOCAB  = SPECIAL + PITCH_TOKENS + TIME_TOKENS + DUR_TOKENS
tok2id = {t: i for i, t in enumerate(VOCAB)}
PAD_ID = tok2id["[PAD]"]

NOTE_BASE = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)

def pitch_to_midi(txt: str) -> int:
    m = re.match(r"([A-Ga-g])([#b-]?)(-?\d+)$", txt.strip())
    if not m: return 0
    root, acc, octv = m.groups()
    n = NOTE_BASE[root.upper()]
    if acc in {"#", "♯"}: n += 1
    elif acc in {"b", "-", "♭"}: n -= 1
    midi = (int(octv) + 1) * 12 + n
    return max(0, min(127, midi))

def ms_bucket(ms: float) -> int:
    return min(MAX_TICK, int(round(ms / RES_MS)))

def explode(js: str):
    seq = [tok2id["[START_SEQ]"]]
    for tok in json.loads(js):
        if not tok.startswith("[NOTE]"): continue
        parts = tok.split()
        pitch  = parts[1].split(":")[1][:-1]
        start  = float(parts[2].split(":")[1][:-1])
        dur    = float(parts[4].split(":")[1][:-2])
        seq += [
            tok2id["[NOTE]"],
            tok2id[f"P_{pitch_to_midi(pitch)}"],
            tok2id[f"T_{ms_bucket(start*1000)}"],
            tok2id[f"DUR_{ms_bucket(dur*1000)}"]
        ]
    seq.append(tok2id["[END_SEQ]"])
    return seq[:SEQ_LEN]

# ── dataset ───────────────────────────────────────────────────────────────
dfs = [pd.read_csv(p, nrows=MAX_ROWS) for p in glob.glob(CSV_GLOB)]
df  = pd.concat(dfs, ignore_index=True)

class MidiDS(Dataset):
    def __init__(self, series):
        self.data = [explode(js) for js in tqdm(series, desc="tokenising")]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        s += [PAD_ID]*(SEQ_LEN - len(s))
        return torch.tensor(s[:-1]), torch.tensor(s[1:])

dl = DataLoader(MidiDS(df["tokens"]), batch_size=BATCH,
                shuffle=True, pin_memory=True, num_workers=0)

# ── model ─────────────────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL))
        blk      = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL*4,
                                              batch_first=True)
        self.tr  = nn.TransformerEncoder(blk, N_LAYER)
        self.fc  = nn.Linear(D_MODEL, V)
    def forward(self, x):
        x = self.emb(x) + self.pos[:x.size(1)]
        return self.fc(self.tr(x))

model, opt = GPT(len(VOCAB)).to(DEVICE), torch.optim.AdamW(
    params=GPT(len(VOCAB)).parameters() if False else model.parameters(), lr=LR
)  # trick to avoid mypy warning
opt = torch.optim.AdamW(model.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss(ignore_index=PAD_ID)

def save(tag):
    torch.save({"model": model.state_dict(),
                "vocab": tok2id,
                "cfg": dict(seq_len=SEQ_LEN, d_model=D_MODEL,
                            n_head=N_HEAD, n_layer=N_LAYER,
                            res_ms=RES_MS, max_tick=MAX_TICK)},
               f"ckpt_{tag}.pt")

# ── train ─────────────────────────────────────────────────────────────────
last, interval = time.time(), SAVE_HOURS*3600
for ep in range(1, EPOCHS+1):
    pbar = tqdm(dl, desc=f"epoch {ep}/{EPOCHS}")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = lossf(model(x).view(-1, len(VOCAB)), y.view(-1))
        loss.backward(); opt.step()
        pbar.set_postfix(loss=float(loss))
        if time.time() - last >= interval:
            save(f"ep{ep}_t{int(time.time())}")
            last = time.time()
    save(f"ep{ep}")
save("final")
print("✓ training complete")
