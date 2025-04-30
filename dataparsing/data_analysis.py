import csv
from collections import Counter
import ast
import sys

csv.field_size_limit(sys.maxsize)

key_signature_counter = Counter()
instrument_counter = Counter()

with open("lmd_full.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 20000:
            break

        key_signature = row["key_signature"]
        key_signature_counter[key_signature] += 1

        try:
            tokens = ast.literal_eval(row["tokens"])
            for token in tokens:
                if token.startswith("[INSTRUMENT]"):
                    instrument = token.replace("[INSTRUMENT] ", "")
                    instrument_counter[instrument] += 1
        except:
            print(f"Skipping malformed tokens in file: {row['file']}")

with open("analysis_output.txt", "w", encoding="utf-8") as out:
    out.write("Key Signature Counts:\n")
    for key, count in sorted(key_signature_counter.items(), key=lambda x: x[1], reverse=True):
        if count > 1:
            out.write(f"{key}: {count}\n")

    out.write("\nInstrument Counts:\n")
    for instrument, count in sorted(instrument_counter.items(), key=lambda x: x[1], reverse=True):
        if count > 1:
            out.write(f"{instrument}: {count}\n")
