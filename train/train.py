"""
train_demo_fixed.py
Minimal end-to-end smoke-test: learn to predict the next music token
conditioned on BPM / Key / Instrument tokens that are already present
inside each sequence.

‣ Expects one or more CSV files whose “tokens” column is a JSON list.
‣ Pads / truncates every sequence to SEQ_LEN so the DataLoader can stack.
‣ Uses a tiny batch-first causal Transformer so it runs on a laptop GPU / CPU.
"""

import glob, json
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------------------------------------------------------
CSV_GLOB   = "lmd_full.csv"     # change if your CSV is elsewhere
READ_ROWS  = 1000             # quick test; raise when things work
SEQ_LEN    = 256                # max tokens per sample (incl. [PAD])
BATCH_PHYS = 4                  # small physical batch
ACC_STEPS  = 4                  # gradient-accumulation => logical batch = 16
EPOCHS     = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------------------------

# 1) Load a subset of rows ----------------------------------------------------
dfs = [pd.read_csv(p, nrows=READ_ROWS) for p in glob.glob(CSV_GLOB)]
df  = pd.concat(dfs, ignore_index=True)
df["tokens"] = df["tokens"].apply(json.loads)

# 2) Build vocabulary ---------------------------------------------------------
vocab_tokens = {tok for seq in df["tokens"] for tok in seq}
PAD_TOKEN = "[PAD]"
vocab_tokens.add(PAD_TOKEN)

tok2id = {t:i for i,t in enumerate(sorted(vocab_tokens))}
id2tok = {i:t for t,i in tok2id.items()}
PAD_ID = tok2id[PAD_TOKEN]

def encode(seq):
    """map string tokens ➜ integer ids and cut to SEQ_LEN"""
    return [tok2id[t] for t in seq][:SEQ_LEN]

# 3) Dataset & DataLoader -----------------------------------------------------
class MidiTokenDataset(Dataset):
    def __init__(self, sequences):
        self.data = [encode(s) for s in sequences]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]

        # pad up to SEQ_LEN with PAD_ID
        padded = seq + [PAD_ID]*(SEQ_LEN - len(seq))

        # causal LM: x is everything except last token, y is everything except first
        x = torch.tensor(padded[:-1])        # length = SEQ_LEN-1
        y = torch.tensor(padded[1:])         # length = SEQ_LEN-1
        return x, y

dataset = MidiTokenDataset(df["tokens"])
loader  = DataLoader(dataset,
                     batch_size=BATCH_PHYS,
                     shuffle=True,
                     num_workers=0)          # keep 0 on Windows/macOS

# 4) Mini causal Transformer --------------------------------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = nn.Parameter(torch.zeros(SEQ_LEN-1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model*4,
            batch_first=True   # suppress nested-tensor warning
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):                     # x shape: [B, S]
        x = self.embed(x) + self.pos[:x.size(1)]
        x = self.tr(x)                        # [B, S, d_model]
        return self.fc(x)                     # [B, S, vocab]

model = MiniGPT(len(tok2id)).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=5e-4)
lossf = nn.CrossEntropyLoss(ignore_index=PAD_ID)

# 5) Training loop ------------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(loader, desc=f"epoch {epoch+1}")
    opt.zero_grad()
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)                         # [B, S, V]
        loss = lossf(logits.view(-1, logits.size(-1)),
                     y.view(-1)) / ACC_STEPS

        loss.backward()

        if (step + 1) % ACC_STEPS == 0 or (step + 1) == len(loader):
            opt.step()
            opt.zero_grad()

        pbar.set_postfix(loss=float(loss * ACC_STEPS))

print("✓ training finished")
torch.save({"model": model.state_dict(), "vocab": tok2id},
           "demo_checkpoint.pt")
