import pretty_midi
from music21 import converter, stream, note, scale



midi_file = pretty_midi.PrettyMIDI('80df1867935371808ab60eabdad2a1d2.mid')
tempo = midi_file.get_tempo_changes()[1][0]
print(f"BPM: {tempo}")

score = converter.parse('80df1867935371808ab60eabdad2a1d2.mid')
key_signature = score.analyze('key')
print(f"Key Signature: {key_signature}")

for i, instr in enumerate(midi_file.instruments):
    name = instr.name or pretty_midi.program_to_instrument_name(instr.program)
    print(f"Track {i+1}: {name} (Program {instr.program}, Channel {instr.program % 16})")

note_pitches = [note.pitch for note in midi_file.instruments[0].notes]
note_durations = [note.end - note.start for note in midi_file.instruments[0].notes]
# print(f"Note Pitches: {note_pitches}")
note_names = [pretty_midi.note_number_to_name(note) for note in note_pitches]
print(f"Note Names: {note_names}")

# s = stream.Stream()
# for p in set(note_pitches):
#     s.append(note.Note(p))

# possible_scales = [
#     scale.MajorScale(),
#     scale.MinorScale(),
#     scale.HarmonicMinorScale(),
#     scale.MelodicMinorScale(),
#     # scale.MajorPentatonicScale(),
#     # scale.MinorPentatonicScale(),
#     scale.DorianScale(),
#     scale.PhrygianScale(),
# ]

# best_scale = None
# max_matches = 0

# for scale in possible_scales:
#     scale_pitches = [p.nameWithOctave for p in scale.getPitches(scale.tonic.nameWithOctave + '4', scale.tonic.nameWithOctave + '5')]
#     # match = sum(1 for n in s.notes if n.name in scale.getPitches(scale.tonic.name, scale.tonic.name + '8'))
#     match = sum(1 for n in s.notes if n.nameWithOctave in scale_pitches or n.name in scale_pitches)
#     if match > max_matches:
#         max_matches = match
#         best_scale = scale

# if best_scale:
#     print(f"Best matching scale: {best_scale.name} in {best_scale.tonic.name}")
# else:
#     print("No matching scale found.")