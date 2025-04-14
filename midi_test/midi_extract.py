import pretty_midi
from music21 import converter, stream, note, scale

# midi_file = pretty_midi.PrettyMIDI('80df1867935371808ab60eabdad2a1d2.mid')
def extract_data(midi_file):
    midi_file = pretty_midi.PrettyMIDI(midi_file)
    tempo = midi_file.get_tempo_changes()[1][0]
    print(f"BPM: {tempo}")

    score = converter.parse('80df1867935371808ab60eabdad2a1d2.mid')
    key_signature = score.analyze('key')
    print(f"Key Signature: {key_signature}")

    instruments = {}

    for i, instr in enumerate(midi_file.instruments):
        name = instr.name or pretty_midi.program_to_instrument_name(instr.program)
        note_infos = [{
            "name": pretty_midi.note_number_to_name(note.pitch),
            "start": round(note.start, 3),
            "end": round(note.end, 3),
            "duration": round(note.end - note.start, 3)
        } for note in instr.notes]


        if name in instruments:
            instruments[name].extend(note_infos)
        else:
            instruments[name] = note_infos


    # print(f"Number of instruments: {len(instruments)}")
    # for name, notes in instruments.items():
        # print(f"Instrument: {name}")
        # for note in notes:
            # print(f"  {note['name']}: Start={note['start']}, End={note['end']}, Duration={note['duration']}")
    # note_durations = [note.end - note.start for note in midi_file.instruments[0].notes]
    midi_data = {
        "BPM": tempo,
        "Key Signature": str(key_signature),
        "Instruments": instruments
    }
    return midi_data