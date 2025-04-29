"""
train_large.py  –  full-scale causal LM for Lakh MIDI Dataset tokens
────────────────────────────────────────────────────────────────────
• Dataset: one or many CSVs with a “tokens” JSON column
• Model  : GPT-style Transformer (configurable depth / width)
• Scales automatically across GPUs / BF16 / FP16 via 🤗 Accelerate
"""

from pathlib import Path
import json, pandas as pd, torch, torch.nn as nn
from torch.utils.data import IterableDataset
from accelerate import Accelerator
from tqdm.auto import tqdm

# ───────────────────────────── hyper-parameters ────────────────────────────
CFG = dict(
    CSV_GLOB      = "lmd_mini.csv",   # glob with ** for nested dirs
    SEQ_LEN       = 512,
    BATCH_PHYS    = 4,        # physical batch / device
    ACC_STEPS     = 8,        # logical batch = BATCH_PHYS*ACC_STEPS (32 here)
    EPOCHS        = 5,
    LR            = 3e-4,
    D_MODEL       = 768,
    N_HEAD        = 12,
    N_LAYER       = 12,
    SAVE_EVERY    = 5_000,    # steps
    OUT_DIR       = Path("checkpoints_large"),
)
# ───────────────────────────────────────────────────────────────────────────

# 1)  build vocabulary only once (streaming over first CSV) -----------------
def yield_tokens(csv_path):
    for chunk in pd.read_csv(csv_path, usecols=["tokens"], chunksize=10_000):
        for js in chunk["tokens"]:
            try:
                yield from json.loads(js)
            except Exception:
                pass

csv_files = list(Path().glob(CFG["CSV_GLOB"]))
vocab = {"[PAD]"}
for f in csv_files:
    vocab.update(yield_tokens(f))
tok2id = {t:i for i,t in enumerate(sorted(vocab))}
PAD_ID  = tok2id["[PAD]"]

print(f"✓ vocab size: {len(tok2id):,}")

# 2)  iterable dataset (stream‐friendly) -------------------------------------
class CSVTokenDataset(IterableDataset):
    def __init__(self, files):
        super().__init__()
        self.files = files
    def parse(self, js):
        try:
            seq = [tok2id[t] for t in json.loads(js)][:CFG["SEQ_LEN"]]
            pad = [PAD_ID]*(CFG["SEQ_LEN"]-len(seq))
            full= seq+pad
            return torch.tensor(full[:-1]), torch.tensor(full[1:])
        except Exception:
            return None
    def __iter__(self):
        for path in self.files:
            for chunk in pd.read_csv(path, usecols=["tokens"],
                                      chunksize=10_000):
                for js in chunk["tokens"]:
                    item = self.parse(js)
                    if item: yield item

dataset = CSVTokenDataset(csv_files)

# 3)  model ------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        S, D = CFG["SEQ_LEN"], CFG["D_MODEL"]
        self.emb  = nn.Embedding(vocab, D)
        self.pos  = nn.Parameter(torch.zeros(S-1, D))
        block     = nn.TransformerEncoderLayer(
                        D, CFG["N_HEAD"], D*4, batch_first=True)
        self.tr   = nn.TransformerEncoder(block, CFG["N_LAYER"])
        self.fc   = nn.Linear(D, vocab)
    def forward(self, x):
        return self.fc(self.tr(self.emb(x)+self.pos[:x.size(1)]))

model = GPT(len(tok2id))

# 4)  training setup with Accelerate ----------------------------------------
accelerator = Accelerator(gradient_accumulation_steps=CFG["ACC_STEPS"],
                          mixed_precision="bf16" if torch.cuda.is_available()
                                          else "no")
optim  = torch.optim.AdamW(model.parameters(), lr=CFG["LR"])
loss_f = nn.CrossEntropyLoss(ignore_index=PAD_ID)

loader = torch.utils.data.DataLoader(dataset,
            batch_size=CFG["BATCH_PHYS"], num_workers=4, pin_memory=True)

model, optim, loader = accelerator.prepare(model, optim, loader)
CFG["OUT_DIR"].mkdir(exist_ok=True)

# 5)  train loop -------------------------------------------------------------
step, epoch = 0, 0
for epoch in range(CFG["EPOCHS"]):
    pbar = tqdm(loader, disable=not accelerator.is_main_process,
                desc=f"epoch {epoch+1}")
    for x, y in pbar:
        with accelerator.accumulate(model):
            logits = model(x)
            loss   = loss_f(logits.view(-1, logits.size(-1)), y.view(-1))
            accelerator.backward(loss)
            optim.step(); optim.zero_grad()

        step += 1
        if accelerator.is_main_process:
            pbar.set_postfix(loss=float(loss))

            if step % CFG["SAVE_EVERY"] == 0:
                name = CFG["OUT_DIR"]/f"step{step:07d}.pt"
                torch.save(
                    {"model": accelerator.get_state_dict(model),
                     "vocab": tok2id}, name)
                print("💾 saved", name)

# final save
if accelerator.is_main_process:
    torch.save({"model": accelerator.get_state_dict(model),
                "vocab": tok2id},
               CFG["OUT_DIR"]/ "final.pt")
    print("✓ training complete")
