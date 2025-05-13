# train_large_autosave.py
import glob, json, time, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

CSV_GLOB = "lmd_full.csv"
MAX_ROWS = 100_000
SEQ_LEN  = 512
D_MODEL  = 512
N_HEAD   = 8
N_LAYER  = 6
BATCH    = 16
EPOCHS   = 6
LR       = 3e-4
SAVE_EVERY_HOURS = 2        # how often to checkpoint
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

dfs = [pd.read_csv(p, nrows=MAX_ROWS) for p in glob.glob(CSV_GLOB)]
df  = pd.concat(dfs, ignore_index=True)
df["tokens"] = df["tokens"].apply(json.loads)

vocab   = {tok for seq in df["tokens"] for tok in seq}
tok2id  = {t:i for i,t in enumerate(sorted(vocab))}
PAD_ID  = len(tok2id)
tok2id["[PAD]"] = PAD_ID

def encode(seq): return [tok2id[t] for t in seq][:SEQ_LEN]

class MidiDS(Dataset):
    def __init__(self, seqs):
        self.data = [encode(s) for s in seqs]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        seq = self.data[i]
        seq += [PAD_ID]*(SEQ_LEN-len(seq))
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])

dl = DataLoader(MidiDS(df["tokens"]), batch_size=BATCH,
                shuffle=True, pin_memory=True, num_workers=0)

class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN-1, D_MODEL))
        blk     = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL*4,
                                             batch_first=True)
        self.tr = nn.TransformerEncoder(blk, N_LAYER)
        self.fc = nn.Linear(D_MODEL, V)
    def forward(self, x):
        x = self.emb(x) + self.pos[:x.size(1)]
        return self.fc(self.tr(x))

model = GPT(len(tok2id)).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss(ignore_index=PAD_ID)

def save(name="latest"):
    torch.save({"model": model.state_dict(),
                "vocab": tok2id,
                "hparams": dict(seq_len=SEQ_LEN, d_model=D_MODEL,
                                n_head=N_HEAD, n_layer=N_LAYER)},
               f"ckpt_{name}.pt")

last_save = time.time()
interval  = SAVE_EVERY_HOURS * 3600

for ep in range(1, EPOCHS+1):
    pbar = tqdm(dl, desc=f"epoch {ep}")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = lossf(model(x).view(-1, len(tok2id)), y.view(-1))
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=float(loss))
        if time.time() - last_save >= interval:
            save(f"ep{ep}_t{int(time.time())}")
            last_save = time.time()
    save(f"ep{ep}")

save("final")
print("✓ training complete")
