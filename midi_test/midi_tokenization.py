from midi_extract import extract_data
def midi_tokenize(midi_file):
    midi_data = extract_data(midi_file)
    tokens = ["[START_SEQUENCE]"]

    tokens.append(f"[BPM] {midi_data['BPM']}")
    tokens.append(f"[KEY_SIGNATURE] {midi_data['Key Signature']}")

    for instrument, notes in midi_data['Instruments'].items():
        tokens.append(f"[INSTRUMENT] {instrument}")
        for note in notes:
            # tokens.append(f"[NOTE] {note['name']} {note['start']} {note['end']} {note['duration']}")   
            tokens.append(f"[NOTE] [PITCH:{note['name']}] [START:{note['start']}] [END:{note['end']}] [DURATION:{note['duration']}]")

    tokens.append("[END_SEQUENCE]")

    return tokens

print(midi_tokenize('80df1867935371808ab60eabdad2a1d2.mid'))