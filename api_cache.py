from fastapi import FastAPI, Form
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
import re

sys.modules['fluidsynth'] = types.ModuleType('fluidsynth')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

CKPT = Path("generate_music/music_generator.pt")
# CKPT = Path("train/ckpt_folders/ckpt_10k4/latest.pt")
# CKPT = Path("train/ckpt_folders/ckpt_10k3/latest.pt")
# CKPT = Path("train/ckpt_folders/ckpt_10k/latest.pt")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
layers = [k for k in ckpt["model"].keys() if "tr.layers." in k]
n_layers = max(int(k.split(".")[2]) for k in layers) + 1
print(f"pytorch model has {n_layers} layers.")
tok2id = ckpt["vocab"]
id2tok = {i: t for t, i in tok2id.items()}
SEQ_LEN = ckpt["model"]["pos"].shape[0]
D_MODEL = ckpt["model"]["pos"].shape[1]

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, layer_past=None):
        """
        x: [B, T_new, C]
        layer_past: tuple of (past_keys, past_values), each [B, T_past, C]
        returns:
          out: [B, T_new, C]
          present: (all_keys, all_values) each [B, T_past+T_new, C]
        """
        # 1) self‑attn
        x_norm = self.ln1(x)
        # we want q against k/v
        q = k = v = x_norm
        if layer_past is not None:
            past_k, past_v = layer_past
            # concatenate along time dim
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        # update cache
        present = (k, v)
        # 2) residual + feed‑forward
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present

class GPTWithKV(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_head, n_layer):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(seq_len, d_model))
        self.layers = nn.ModuleList([
            GPTBlock(d_model, n_head, d_model * 4)
            for _ in range(n_layer)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, idx, past_kv=None):
        """
        idx: [B, T] token IDs
        past_kv: list of length n_layer, each either None or (B, T_past, C)x2
        returns:
          logits: [B, T, V]
          presents: new list of (k,v) per layer
        """
        B, T = idx.size()
        if past_kv is None:
            past_kv = [None] * len(self.layers)

        x = self.tok_emb(idx) + self.pos_emb[:T]
        presents = []
        for layer, past in zip(self.layers, past_kv):
            x, present = layer(x, past)
            presents.append(present)

        logits = self.head(x)
        return logits, presents

model = GPTWithKV(
    vocab_size=len(tok2id),
    seq_len=SEQ_LEN,
    d_model=D_MODEL,
    n_head=8,
    n_layer=n_layers,
)
# model = GPT(len(tok2id), SEQ_LEN, D_MODEL, n_head=8, n_layer=6)
# model = GPT(len(tok2id), SEQ_LEN, D_MODEL, n_head=8, n_layer=4)

def remap_state_dict(old_sd):
    new_sd = {}
    for k, v in old_sd.items():
        # emb → tok_emb
        k2 = k.replace("emb.weight", "tok_emb.weight")
        # pos → pos_emb
        k2 = k2.replace("pos", "pos_emb")
        # fc → head
        k2 = k2.replace("fc.", "head.")
        # transformer layers → GPTBlock layers
        k2 = re.sub(r"tr\.layers\.(\d+)\.self_attn",   r"layers.\1.attn",  k2)
        k2 = re.sub(r"tr\.layers\.(\d+)\.norm1",        r"layers.\1.ln1",   k2)
        k2 = re.sub(r"tr\.layers\.(\d+)\.norm2",        r"layers.\1.ln2",   k2)
        k2 = re.sub(r"tr\.layers\.(\d+)\.linear1",      r"layers.\1.mlp.0", k2)
        k2 = re.sub(r"tr\.layers\.(\d+)\.linear2",      r"layers.\1.mlp.2", k2)
        new_sd[k2] = v
    return new_sd

sd = remap_state_dict(ckpt["model"])
model.load_state_dict(sd)
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
def sample_kvcache(model, prompt, max_len=512, temperature=1.0, top_k=50, device="cpu"):
    model.to(device).eval()
    input_ids = torch.tensor([tok2id[t] for t in prompt], device=device).unsqueeze(0)
    logits, past_kv = model(input_ids)
    generated = input_ids

    for _ in range(max_len - input_ids.size(1)):
        last_id = generated[:, -1:].to(device)
        logits, past_kv = model(last_id, past_kv)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            vals, idxs = logits.topk(top_k)
            mask = torch.full_like(logits, -1e10)
            mask.scatter_(1, idxs, 0.0)
            logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_id], dim=1)

        if next_id.item() == tok2id.get("[END_SEQUENCE]", -1):
            break

    return [id2tok[i] for i in generated.squeeze().tolist()]

@app.post("/generate")
def generate_music(prompt: str = Form(...)):
    print(f"\nPrompt received: {prompt}")
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
    tokens = sample_kvcache(model, gen_prompt, max_len=SEQ_LEN, temperature=1.0, top_k=50, device="cpu")

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

    midi_bytes = BytesIO()
    pm.write(midi_bytes)

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
        os.remove(midi_path)
