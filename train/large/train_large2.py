# train_large_compact.py  –  memory-friendly Transformer for music tokens
import glob, json, time, re, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─────────────── configuration ────────────────────────────────────────────
CSV_GLOB        = "lmd_full.csv"   # path or pattern
MAX_ROWS        = 10000          # None ⇒ all rows
SEQ_LEN         = 512
D_MODEL         = 512
N_HEAD, N_LAYER = 8, 6
BATCH           = 16
EPOCHS          = 6
LR              = 3e-4
SAVE_HOURS      = 2
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

RES_MS, MAX_TICK = 50, 4095        # 50 ms resolution, ticks 0-4095
# ───────────────────────────────────────────────────────────────────────────

# ─────────────── vocabulary ───────────────────────────────────────────────
SPECIAL       = ["[PAD]", "[START_SEQ]", "[END_SEQ]", "[NOTE]"]
PITCH_TOKENS  = [f"P_{i}"      for i in range(128)]          # MIDI pitches
TIME_TOKENS   = [f"T_{i}"      for i in range(MAX_TICK + 1)]
DUR_TOKENS    = [f"DUR_{i}"    for i in range(MAX_TICK + 1)]

VOCAB  = SPECIAL + PITCH_TOKENS + TIME_TOKENS + DUR_TOKENS
tok2id = {t: i for i, t in enumerate(VOCAB)}
PAD_ID = tok2id["[PAD]"]

NOTE_BASE = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)

def pitch_to_midi(txt: str) -> int:
    m = re.match(r"([A-Ga-g])([#b-]?)(-?\d+)$", txt.strip())
    if not m:                                   # fallback to middle C
        return 60
    root, acc, octv = m.groups()
    semitone = NOTE_BASE[root.upper()]
    if acc in {"#", "♯"}: semitone += 1
    elif acc in {"b", "-", "♭"}: semitone -= 1
    midi = (int(octv) + 1) * 12 + semitone      # MIDI convention
    return max(0, min(127, midi))

def bucket(ms: float) -> int:
    return min(MAX_TICK, int(round(ms / RES_MS)))

def explode(js: str):
    seq = [tok2id["[START_SEQ]"]]
    for tok in json.loads(js):
        if not tok.startswith("[NOTE]"):
            continue
        parts   = tok.split()
        pitch_s = parts[1].split(":")[1][:-1]
        start   = float(parts[2].split(":")[1][:-1])
        dur     = float(parts[4].split(":")[1][:-2])

        seq += [
            tok2id["[NOTE]"],
            tok2id[f"P_{pitch_to_midi(pitch_s)}"],
            tok2id[f"T_{bucket(start*1000)}"],
            tok2id[f"DUR_{bucket(dur*1000)}"]
        ]
    seq.append(tok2id["[END_SEQ]"])
    return seq[:SEQ_LEN]

# ─────────────── dataset ──────────────────────────────────────────────────
dfs = [pd.read_csv(p, nrows=MAX_ROWS) for p in glob.glob(CSV_GLOB)]
df  = pd.concat(dfs, ignore_index=True)

class MidiDataset(Dataset):
    def __init__(self, token_series):
        self.data = [explode(js) for js in tqdm(token_series, desc="tokenising")]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        s = self.data[idx] + [PAD_ID]*(SEQ_LEN - len(self.data[idx]))
        return torch.tensor(s[:-1]), torch.tensor(s[1:])

loader = DataLoader(MidiDataset(df["tokens"]), batch_size=BATCH,
                    shuffle=True, pin_memory=True, num_workers=0)

# ─────────────── model ────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL))
        block    = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL*4,
                                              batch_first=True)
        self.tr  = nn.TransformerEncoder(block, N_LAYER)
        self.fc  = nn.Linear(D_MODEL, vocab_size)
    def forward(self, x):
        x = self.emb(x) + self.pos[:x.size(1)]
        return self.fc(self.tr(x))

model = GPT(len(VOCAB)).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss(ignore_index=PAD_ID)

def save(tag: str):
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": tok2id,
            "cfg": dict(seq_len=SEQ_LEN, d_model=D_MODEL,
                        n_head=N_HEAD, n_layer=N_LAYER,
                        res_ms=RES_MS, max_tick=MAX_TICK)
        },
        f"ckpt_{tag}.pt"
    )

# ─────────────── training loop ────────────────────────────────────────────
last_save = time.time()
interval  = SAVE_HOURS * 3600

for ep in range(1, EPOCHS + 1):
    pbar = tqdm(loader, desc=f"epoch {ep}/{EPOCHS}")
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

    save(f"ep{ep}")

save("final")
print("✓ training complete")
