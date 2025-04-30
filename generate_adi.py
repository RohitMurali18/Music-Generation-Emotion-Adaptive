from emotion_analysis import inference, EATS
import torch, pretty_midi, re
from pathlib import Path
import sys, types
import shutil

# Inject dummy fluidsynth if needed
sys.modules['fluidsynth'] = types.ModuleType('fluidsynth')

# === 1. Get prediction & mapping ===
# prompt = "i am walking down a road and i see a rainbow and it is sunny. i love life."
prompt = input("Enter a description or feeling: ")
allpredictions = inference.predict(prompt)
mapping = EATS.get_music_params(allpredictions)
print("🎵 Music Mapping:", mapping)

# === 2. Load model checkpoint & set up model ===
CKPT = Path("generate_music/demo_checkpoint.pt")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)

tok2id = ckpt["vocab"]
id2tok = {i: t for t, i in tok2id.items()}
PAD_ID = tok2id["[PAD]"]
# print([t for t in tok2id if t.startswith("[KEY_SIGNATURE]")])
SEQ_LEN = ckpt["model"]["pos"].shape[0] + 1
D_MODEL = ckpt["model"]["pos"].shape[1]

import torch.nn as nn
class GPT(nn.Module):
    def __init__(self, vocab, seq_len, d_model, n_head=4, n_layer=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(seq_len-1, d_model))
        block = nn.TransformerEncoderLayer(d_model, n_head, d_model*4, batch_first=True)
        self.tr = nn.TransformerEncoder(block, n_layer)
        self.fc = nn.Linear(d_model, vocab)
    def forward(self, x):
        return self.fc(self.tr(self.emb(x) + self.pos[:x.size(1)]))

model = GPT(len(tok2id), SEQ_LEN, D_MODEL)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

# === 3. Helper functions ===
def encode(tokens): return torch.tensor([tok2id[t] for t in tokens])
def decode(ids): return [id2tok[int(i)] for i in ids]

@torch.no_grad()
def sample(prompt, max_len=512, temperature=1.0, top_k=50, device="cpu"):
    ids = encode(prompt).unsqueeze(0).to(device)
    model.to(device)
    for _ in range(max_len - len(prompt)):
        logits = model(ids)[:, -1, :] / temperature
        if top_k:
            topk_idx = logits.topk(top_k).indices
            mask = torch.full_like(logits, -1e10).scatter_(1, topk_idx, 0)
            logits = logits + mask
        probs = torch.softmax(logits, -1)
        next_i = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_i], dim=1)
        if next_i.item() == tok2id.get("[END_SEQUENCE]", -1):
            break
    return decode(ids.squeeze())
def normalize_key_signature(key_string):
    key_string = key_string.replace("♭", "-").replace("♯", "#")
    parts = key_string.strip().split()
    if len(parts) == 2:
        key, scale = parts
        return f"[KEY_SIGNATURE] {key} {scale.lower()}"
    return f"[KEY_SIGNATURE] {key_string}"
def closest_bpm_token(val):
    bpm_toks = [t for t in tok2id if t.startswith("[BPM]")]
    return min(bpm_toks, key=lambda s: abs(float(s.split()[-1]) - val))

# === 4. Use `mapping` directly to generate ===
bpm_tok = closest_bpm_token(mapping["bpm"])
key = mapping["key"]

FAMILY_TO_INSTRUMENTS = {
    "Strings": ["Violin"],
    "Piano": ["Acoustic Grand Piano"],
    "Woodwind": ["Flute"]
}

instr = []
for fam in mapping["all_families"]:
    instr.extend(FAMILY_TO_INSTRUMENTS.get(fam, []))

key_token = normalize_key_signature(mapping["key"])
gen_prompt = ["[START_SEQUENCE]", bpm_tok, key_token] + \
             [f"[INSTRUMENT] {i}" for i in instr]

tokens = sample(gen_prompt, max_len=SEQ_LEN)
print("🎶 Generated token snippet:", tokens[:40], "...")

# === 5. Convert tokens to MIDI ===
note_re = re.compile(r"\[NOTE\] \[PITCH:(.+?)\] \[START:(.+?)\] \[END:(.+?)\] \[DURATION:(.+?)\]")

pm, current_inst = pretty_midi.PrettyMIDI(), None

for tok in tokens:
    if tok.startswith("[INSTRUMENT]"):
        name = tok.split("]",1)[1].strip()
        prog = pretty_midi.instrument_name_to_program(name) \
               if name in pretty_midi.INSTRUMENT_MAP else 0
        current_inst = pretty_midi.Instrument(program=prog, name=name)
        pm.instruments.append(current_inst)
    elif (m := note_re.match(tok)) and current_inst:
        pitch = pretty_midi.note_name_to_number(m.group(1))
        start, end = float(m.group(2)), float(m.group(3))
        current_inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        )

out_path = Path("generate_music/generated.mid")
pm.write(str(out_path))
target_path = Path("frontend/public/generated.mid")
shutil.copy(out_path, target_path)
print("✅ Copied MIDI to frontend/public/generated.mid")
print("✅ Final MIDI saved to:", out_path.resolve())