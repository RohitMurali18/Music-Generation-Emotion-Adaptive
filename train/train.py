# train_demo.py
import ast, json, glob, random
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

CSV_GLOB = "lmd_mini.csv"   # adjust
MAX_ROWS  = 10_000          # small subset for quick test
SEQ_LEN   = 512             # truncate / pad length
BATCH     = 8
EPOCHS    = 5
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# 1) read a handful of rows ---------------------------------------------------
dfs = [pd.read_csv(p, nrows=MAX_ROWS) for p in glob.glob(CSV_GLOB)]
df  = pd.concat(dfs, ignore_index=True)

# parse json string of tokens -> python list
df["tokens"] = df["tokens"].apply(lambda s: json.loads(s))

# 2) build vocab --------------------------------------------------------------
all_tokens = {tok for seq in df["tokens"] for tok in seq}
tok2id = {t:i for i,t in enumerate(sorted(all_tokens))}
id2tok = {i:t for t,i in tok2id.items()}

def encode(seq):
    return [tok2id[t] for t in seq][:SEQ_LEN]

PAD_ID = len(tok2id)
tok2id["[PAD]"] = PAD_ID
id2tok[PAD_ID]   = "[PAD]"

# 3) dataset ------------------------------------------------------------------
class MidiTokenDS(Dataset):
    def __init__(self, sequences):
        self.data = [encode(s) for s in sequences]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        pad = [PAD_ID]*(SEQ_LEN-len(seq))
        x   = torch.tensor(seq+pad[:-1])      # input
        y   = torch.tensor(seq[1:]+pad)       # next-token target
        return x, y

dataset = MidiTokenDS(df["tokens"].tolist())
loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

# 4) tiny causal Transformer --------------------------------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab, d_model=256, n_head=4, n_layer=2):
        super().__init__()
        self.emb  = nn.Embedding(vocab, d_model)
        self.pos  = nn.Parameter(torch.zeros(SEQ_LEN, d_model))
        encoder   = nn.TransformerEncoderLayer(d_model, n_head, d_model*4)
        self.tr   = nn.TransformerEncoder(encoder, n_layer)
        self.fc   = nn.Linear(d_model, vocab)

    def forward(self, x):
        x = self.emb(x) + self.pos[:x.size(1)]
        x = self.tr(x)
        return self.fc(x)

model = MiniGPT(len(tok2id)).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=3e-4)
lossf = nn.CrossEntropyLoss(ignore_index=PAD_ID)

# 5) training loop ------------------------------------------------------------
for epoch in range(EPOCHS):
    pbar = tqdm(loader, desc=f"epoch {epoch+1}")
    for x,y in pbar:
        x,y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = lossf(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        pbar.set_postfix(loss=float(loss))

print("✓ demo training finished")
torch.save({"model":model.state_dict(),"vocab":tok2id}, "demo_checkpoint.pt")
