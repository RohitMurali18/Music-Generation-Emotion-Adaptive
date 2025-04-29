import sys, types
# inject dummy fluidsynth module
sys.modules['fluidsynth'] = types.ModuleType('fluidsynth')

import os, torch, json, pretty_midi
from pathlib import Path
import re, random

CKPT = "demo_checkpoint.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)

tok2id = ckpt["vocab"]
id2tok = {i: t for t, i in tok2id.items()}
PAD_ID = tok2id["[PAD]"]

# ---------- infer geometry from checkpoint ---------------------------------
pos_shape   = ckpt["model"]["pos"].shape       # e.g. [512, 256]
SEQ_LEN_WT  = pos_shape[0] + 1                 # +1 because we dropped 1 in save
D_MODEL_WT  = pos_shape[1]

print("checkpoint geometry  →  SEQ_LEN:", SEQ_LEN_WT, "  d_model:", D_MODEL_WT)

# ---------- rebuild model with those dims ----------------------------------
import torch.nn as nn
class GPT(nn.Module):
    def __init__(s, vocab, seq_len, d_model, n_head=4, n_layer=2):
        super().__init__()
        s.emb  = nn.Embedding(vocab, d_model)
        s.pos  = nn.Parameter(torch.zeros(seq_len-1, d_model))
        block  = nn.TransformerEncoderLayer(
                    d_model, n_head, d_model*4, batch_first=True)
        s.tr   = nn.TransformerEncoder(block, n_layer)
        s.fc   = nn.Linear(d_model, vocab)
    def forward(s,x):
        return s.fc(s.tr(s.emb(x)+s.pos[:x.size(1)]))

model = GPT(len(tok2id), SEQ_LEN_WT, D_MODEL_WT)
model.load_state_dict(ckpt["model"], strict=True)   # ✅ loads cleanly
model.eval()

# ---------- helper encode / decode -----------------------------------------
def encode(tokens): return torch.tensor([tok2id[t] for t in tokens])
def decode(ids):    return [id2tok[int(i)] for i in ids]

# ---------- sampler ---------------------------------------------------------
@torch.no_grad()
def sample(prompt, max_len=512, temperature=1.0, top_k=50, device="cpu"):
    ids = encode(prompt).unsqueeze(0).to(device)
    model.to(device)
    for _ in range(max_len-len(prompt)):
        logits = model(ids)[:, -1, :] / temperature
        if top_k:
            topk_idx = logits.topk(top_k).indices
            mask = torch.full_like(logits, -1e10).scatter_(1, topk_idx, 0)
            logits = logits + mask
        probs  = torch.softmax(logits, -1)
        next_i = torch.multinomial(probs, 1)
        ids    = torch.cat([ids, next_i], dim=1)
        if next_i.item() == tok2id.get("[END_SEQUENCE]", -1):
            break
    return decode(ids.squeeze())

# ---------- generate --------------------------------------------------------
def closest_bpm_token(val):
    bpm_toks = [t for t in tok2id if t.startswith("[BPM]")]
    return min(bpm_toks, key=lambda s: abs(float(s.split()[-1]) - val))

bpm_target = 180
bpm_tok    = closest_bpm_token(bpm_target)

key   = "A minor"
instr = ["Violin", "Piano"]

prompt = ["[START_SEQUENCE]",
          bpm_tok,                       # ← use it directly, no f-string
          f"[KEY_SIGNATURE] {key}"] + \
         [f"[INSTRUMENT] {i}" for i in instr]

gen = sample(prompt, max_len=SEQ_LEN_WT)
print("generated tokens snippet:", gen[:40], "...")

# ---------- tokens → MIDI ---------------------------------------------------
note_re = re.compile(
    r"\[NOTE\] \[PITCH:(.+?)\] \[START:(.+?)\] \[END:(.+?)\] \[DURATION:(.+?)\]"
)

pm, current_inst = pretty_midi.PrettyMIDI(), None

for tok in gen:
    if tok.startswith("[INSTRUMENT]"):
        name = tok.split("]",1)[1].strip()
        prog = pretty_midi.instrument_name_to_program(name) \
               if name in pretty_midi.INSTRUMENT_MAP else 0
        current_inst = pretty_midi.Instrument(program=prog, name=name)
        pm.instruments.append(current_inst)
        continue
    m = note_re.match(tok)
    if m and current_inst:
        pitch = pretty_midi.note_name_to_number(m.group(1))
        start = float(m.group(2)); end = float(m.group(3))
        current_inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        )

out_midi = Path("generated.mid")
pm.write(str(out_midi))          # ← cast Path ➜ str
print("✅  MIDI saved ->", out_midi.resolve())
