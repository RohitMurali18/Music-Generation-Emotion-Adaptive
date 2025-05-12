import sys, types, torch, re, pretty_midi, random
from pathlib import Path

sys.modules['fluidsynth'] = types.ModuleType('fluidsynth')

CKPT = "ckpt_full/latest.pt"
ckpt = torch.load(CKPT, map_location="cpu")
tok2id = ckpt["vocab"]; id2tok = {i:t for t,i in tok2id.items()}
hp = ckpt["hparams"]; PAD_ID = tok2id["[PAD]"]

class GPT(torch.nn.Module):
    def __init__(self, V, hp):
        super().__init__()
        self.emb = torch.nn.Embedding(V, hp["d_model"])
        self.pos = torch.nn.Parameter(torch.zeros(hp["seq_len"]-1, hp["d_model"]))
        blk = torch.nn.TransformerEncoderLayer(hp["d_model"], hp["n_head"],
                                               hp["d_model"]*4, batch_first=True)
        self.tr = torch.nn.TransformerEncoder(blk, hp["n_layer"])
        self.fc = torch.nn.Linear(hp["d_model"], V)
    def forward(self, x):
        return self.fc(self.tr(self.emb(x) + self.pos[:x.size(1)]))

model = GPT(len(tok2id), hp)
model.load_state_dict(ckpt["model"], strict=True).eval()

def encode(toks): return torch.tensor([tok2id[t] for t in toks])
def decode(ids):  return [id2tok[int(i)] for i in ids]

@torch.no_grad()
def sample(prompt, max_len=hp["seq_len"], temp=1.0, top_k=50, device="cpu"):
    ids = encode(prompt).unsqueeze(0).to(device); model.to(device)
    for _ in range(max_len-len(prompt)):
        logits = model(ids)[:,-1,:] / temp
        if top_k:
            idx = logits.topk(top_k).indices
            mask = torch.full_like(logits, -1e9).scatter_(1, idx, 0)
            logits += mask
        next_id = torch.multinomial(torch.softmax(logits, -1), 1)
        ids = torch.cat([ids, next_id], 1)
        if next_id.item() == tok2id.get("[END_SEQUENCE]", -1): break
    return decode(ids.squeeze())

def closest_bpm(val):
    bpms = [t for t in tok2id if t.startswith("[BPM]")]
    return min(bpms, key=lambda s: abs(float(s.split()[-1])-val))

prompt = ["[START_SEQUENCE]", closest_bpm(180), "[KEY_SIGNATURE] A minor",
          "[INSTRUMENT] Violin", "[INSTRUMENT] Piano"]

gen = sample(prompt)
print(gen[:400], "...")

note_re = re.compile(r"\[NOTE\] \[PITCH:(.+?)\] "
                     r"\[START:(.+?)\] \[END:(.+?)\] \[DURATION:(.+?)\]")
pm, cur = pretty_midi.PrettyMIDI(), None
for tok in gen:
    if tok.startswith("[INSTRUMENT]"):
        name = tok.split("]",1)[1].strip()
        prog = pretty_midi.instrument_name_to_program(name) \
               if name in pretty_midi.INSTRUMENT_MAP else 0
        cur = pretty_midi.Instrument(program=prog, name=name)
        pm.instruments.append(cur); continue
    m = note_re.match(tok)
    if m and cur:
        p,s,e,d = m.groups()
        cur.notes.append(pretty_midi.Note(100,
            pretty_midi.note_name_to_number(p),
            float(s), float(e)))
out = Path("generated.mid"); pm.write(str(out))
print("saved", out.resolve())
