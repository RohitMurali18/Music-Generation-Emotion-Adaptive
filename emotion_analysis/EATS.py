import json
import random
import pandas as pd
from typing import Union, List, Tuple, Dict
import os

LOOKUP_PATH = os.path.join(os.path.dirname(__file__), "lookup_table.csv")
_df = pd.read_csv(LOOKUP_PATH)

EATS = {
    row["emotion"]: {
        "bpm_min": int(row["bpm_min"]),
        "bpm_max": int(row["bpm_max"]),
        "key": row["key"],
        "scale_type": row["scale_type"],
        "instrument_families": json.loads(row["instrument_families"]),
    }
    for _, row in _df.iterrows()
}

def _params_for_label(label: str) -> Dict:
    label_lc = label.lower()
    if label_lc not in EATS:
        raise ValueError(f"Emotion '{label}' not in lookup table")

    entry = EATS[label_lc]
    bpm = random.randint(entry["bpm_min"], entry["bpm_max"])
    inst_family = random.choice(entry["instrument_families"])

    return {
        "emotion": label_lc,
        "bpm": bpm,
        "key": entry["key"],
        "scale_type": entry["scale_type"],
        "inst_family": inst_family,
        "all_families": entry["instrument_families"],
    }

def get_music_params(emotions: Union[str, List[str], Tuple[str, ...]]) -> Union[Dict, List[Dict]]:
    if isinstance(emotions, str):
        return _params_for_label(emotions)
    return [_params_for_label(lab) for lab in emotions]
