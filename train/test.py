# debug_loader.py
import json, pandas as pd
from train_large import CSVStream, tok2id, PAD_ID, SEQ_LEN   # reuse your defs
from torch.utils.data import DataLoader

ds = CSVStream()
loader = DataLoader(ds, batch_size=4, num_workers=0)

for i, (x, y) in enumerate(loader):
    print("got batch", i, x.shape)
    if i == 5:
        break
