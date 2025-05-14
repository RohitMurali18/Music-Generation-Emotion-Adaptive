# train_large_bpm_key.py
import glob, json, re, time, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

CSV_GLOB      = "lmd_full.csv"
MAX_ROWS      = 10_000
SEQ_LEN       = 512
D_MODEL       = 512
N_HEAD        = 8
N_LAYER       = 6
BATCH         = 16
EPOCHS        = 6
LR            = 3e-4
SAVE_HOURS    = 2
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

RES_MS, MAX_TICK = 50, 4095
MIN_BPM, MAX_BPM = 20, 250                 # round-clamp into this range

SPECIAL = ["[PAD]", "[START_SEQ]", "[END_SEQ]", "[NOTE]"]
PITCH_TOKENS = [f"P_{i}" for i in range(128)]
TIME_TOKENS  = [f"T_{i}" for i in range(MAX_TICK + 1)]
DUR_TOKENS   = [f"DUR_{i}" for i in range(MAX_TICK + 1)]
BPM_TOKENS   = [f"BPM_{i}" for i in range(MIN_BPM, MAX_BPM + 1)]
KEY_TOKENS   = [f"KEY_{i}" for i in range(24)]            # 0-11 maj, 12-23 min

VOCAB  = SPECIAL + BPM_TOKENS + KEY_TOKENS + PITCH_TOKENS + TIME_TOKENS + DUR_TOKENS
tok2id = {t: i for i, t in enumerate(VOCAB)}
PAD_ID = tok2id["[PAD]"]

NOTE_BASE = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)

def pitch_to_midi(txt):
    m = re.match(r"([A-Ga-g])([#b-]?)(-?\d+)$", txt.strip())
    if not m: return 60
    r, a, o = m.groups(); s = NOTE_BASE[r.upper()]
    if a in {"#", "♯"}: s += 1
    elif a in {"b", "-", "♭"}: s -= 1
    return max(0, min(127, (int(o)+1)*12 + s))

def key_to_idx(txt):
    m = re.match(r"([A-Ga-g])([#b-]?)[\s_-]*(major|minor)", txt.strip(), re.I)
    if not m: return 0
    r, a, mode = m.groups()
    s = NOTE_BASE[r.upper()]
    if a in {"#", "♯"}: s += 1
    elif a in {"b", "-", "♭"}: s -= 1
    return (s % 12) + (12 if mode.lower()=="minor" else 0)

def bucket(ms): return min(MAX_TICK, int(round(ms / RES_MS)))

def explode(js):
    bpm_tok = key_tok = None
    seq = [tok2id["[START_SEQ]"]]
    for t in json.loads(js):
        if t.startswith("[BPM]"):
            bpm = int(round(float(t.split()[-1])))
            bpm = max(MIN_BPM, min(MAX_BPM, bpm))
            bpm_tok = tok2id[f"BPM_{bpm}"]
        elif t.startswith("[KEY_SIGNATURE]"):
            key_idx = key_to_idx(" ".join(t.split()[1:]))
            key_tok = tok2id[f"KEY_{key_idx}"]
        elif t.startswith("[NOTE]"):
            p = pitch_to_midi(t.split()[1].split(":")[1][:-1])
            parts = t.split()
            s  = float(parts[2].split(":")[1][:-1])
            d  = float(parts[4].split(":")[1][:-2])
            seq += [
                tok2id["[NOTE]"],
                tok2id[f"P_{p}"],
                tok2id[f"T_{bucket(s*1000)}"],
                tok2id[f"DUR_{bucket(d*1000)}"]
            ]
    if bpm_tok: seq.insert(1, bpm_tok)
    if key_tok: seq.insert(2 if bpm_tok else 1, key_tok)
    seq.append(tok2id["[END_SEQ]"])
    return seq[:SEQ_LEN]

dfs = [pd.read_csv(p, nrows=MAX_ROWS) for p in glob.glob(CSV_GLOB)]
df  = pd.concat(dfs, ignore_index=True)

class MidiDS(Dataset):
    def __init__(self, series):
        self.data = [explode(js) for js in tqdm(series, desc="tokenising")]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i] + [PAD_ID]*(SEQ_LEN - len(self.data[i]))
        return torch.tensor(s[:-1]), torch.tensor(s[1:])

loader = DataLoader(MidiDS(df["tokens"]), batch_size=BATCH,
                    shuffle=True, pin_memory=True, num_workers=0)

class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL))
        blk = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL*4, batch_first=True)
        self.tr = nn.TransformerEncoder(blk, N_LAYER)
        self.fc = nn.Linear(D_MODEL, V)
    def forward(self, x):
        x = self.emb(x) + self.pos[:x.size(1)]
        return self.fc(self.tr(x))

model = GPT(len(VOCAB)).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss(ignore_index=PAD_ID)

def save(tag):
    torch.save({"model": model.state_dict(),
                "vocab": tok2id,
                "cfg": dict(seq_len=SEQ_LEN, d_model=D_MODEL,
                            n_head=N_HEAD, n_layer=N_LAYER,
                            res_ms=RES_MS, max_tick=MAX_TICK)},
               f"ckpt_no_inst_{tag}.pt")

last, interval = time.time(), SAVE_HOURS*3600
for ep in range(1, EPOCHS+1):
    pbar = tqdm(loader, desc=f"epoch {ep}/{EPOCHS}")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = lossf(model(x).view(-1, len(VOCAB)), y.view(-1))
        loss.backward(); opt.step()
        pbar.set_postfix(loss=float(loss))
        if time.time()-last >= interval:
            save(f"ep{ep}_t{int(time.time())}"); last = time.time()
    save(f"ep{ep}")
save("final")
print("✓ training complete")
