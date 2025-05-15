from fastapi import FastAPI, Form
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from emotion_analysis import inference, EATS
import torch, pretty_midi, re, os, sys, types
from pathlib import Path
import shutil
from fastapi.responses import StreamingResponse
from midi2audio import FluidSynth
import tempfile
from io import BytesIO
import torch.nn as nn

sys.modules['fluidsynth'] = types.ModuleType('fluidsynth')

app = FastAPI()

# Enable CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_methods=["*"],
    allow_headers=["*"],
)

CKPT = Path("generate_music/demo_checkpoint.pt")
# CKPT = Path("train/ckpt_folders/ckpt_10k4/latest.pt")
# CKPT = Path("train/ckpt_folders/ckpt_10k3/latest.pt")
# CKPT = Path("train/ckpt_folders/ckpt_10k/latest.pt")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
# print(ckpt["model"].keys())
layers = [k for k in ckpt["model"].keys() if "tr.layers." in k]
n_layers = max(int(k.split(".")[2]) for k in layers) + 1
print(f"pytorch model has {n_layers} layers.")
tok2id = ckpt["vocab"]
id2tok = {i: t for t, i in tok2id.items()}
SEQ_LEN = ckpt["model"]["pos"].shape[0] + 1
D_MODEL = ckpt["model"]["pos"].shape[1]

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
# model = GPT(len(tok2id), SEQ_LEN, D_MODEL, n_head=8, n_layer=6)
# model = GPT(len(tok2id), SEQ_LEN, D_MODEL, n_head=8, n_layer=4)
model.load_state_dict(ckpt["model"])
model.eval()

def encode(tokens): return torch.tensor([tok2id[t] for t in tokens])
def decode(ids): return [id2tok[int(i)] for i in ids]
def closest_bpm_token(val):
    bpm_toks = [t for t in tok2id if t.startswith("[BPM]")]
    return min(bpm_toks, key=lambda s: abs(float(s.split()[-1]) - val))
def normalize_key_signature(key_string):
    key_string = key_string.replace("♭", "-").replace("♯", "#")
    parts = key_string.strip().split()
    if len(parts) == 2:
        key, scale = parts
        return f"[KEY_SIGNATURE] {key} {scale.lower()}"
    return f"[KEY_SIGNATURE] {key_string}"
FAMILY_TO_INSTRUMENTS = {
    "Strings": ["Violin"],
    "Piano": ["Acoustic Grand Piano"],
    "Woodwind": ["Flute"]
}
note_re = re.compile(r"\[NOTE\] \[PITCH:(.+?)\] \[START:(.+?)\] \[END:(.+?)\] \[DURATION:(.+?)\]")

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

@app.post("/generate")
def generate_music(prompt: str = Form(...)):
    print(f"\nPrompt received: {prompt}")

    # if 'happy' in prompt:
    #     midi_path = "generate_music/0117612a80915d28b98a8454d5ab0411.mid"
    # elif 'sad' in prompt:
    #     midi_path = "generate_music/0e3872008afc692b86ddf51061592a27.mid"
    # elif 'scare' in prompt:
    #     midi_path = "generate_music/0a5ac59190f0fefa7496df55062b4e8f.mid"
    # elif 'relax' in prompt:
    #     midi_path = "generate_music/0bc5ddea54f25e8cec44968fb1373a02.mid"
    # else: 
    allpredictions = inference.predict(prompt)
    mapping = EATS.get_music_params(allpredictions)

    print("Music Mapping:", mapping)

    bpm_tok = closest_bpm_token(mapping["bpm"])
    key = normalize_key_signature(mapping["key"])
    instruments = []
    for fam in mapping["all_families"]:
        instruments.extend(FAMILY_TO_INSTRUMENTS.get(fam, []))
    print("Instruments:", instruments)
    print("Key:", key)
    print("BPM Token:", bpm_tok)

    gen_prompt = ["[START_SEQUENCE]", bpm_tok, key] + [f"[INSTRUMENT] {i}" for i in instruments]
    tokens = sample(gen_prompt, max_len=SEQ_LEN)

    print("Generated token snippet:", tokens, "...\n")

    pm, current_inst = pretty_midi.PrettyMIDI(), None
    for tok in tokens:
        if tok.startswith("[INSTRUMENT]"):
            name = tok.split("]", 1)[1].strip()
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

    # out_file = Path("frontend/public/generated.mid")
    # pm.write(str(out_file))
    # return FileResponse(out_file, media_type="audio/midi", filename="generated.mid")

    midi_bytes = BytesIO()
    pm.write(midi_bytes)
    # midi_bytes.seek(0)

    # return StreamingResponse(
    #     midi_bytes,
    #     media_type="audio/midi",
    #     headers={"Content-Disposition": "attachment; filename=generated.mid"}
    # )
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midi_file:
        pm.write(midi_file.name)
        midi_path = midi_file.name

    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)

    try:
        fs = FluidSynth('generate_music/FluidR3_GM.sf2')
        fs.midi_to_audio(midi_path, wav_path)

        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename="generated.wav",
        )
    finally:
        # pass
        os.remove(midi_path)