#!/bin/bash
CONFIG_JSON="configs.json"
SCRIPT="config.sh"

# Lire les paramètres depuis le JSON et créer toutes les combinaisons
COMBOS=$(python3 - <<END
import json
from itertools import product

config_json = "$CONFIG_JSON"
script = "$SCRIPT"

with open(config_json) as f:
    config = json.load(f)

keys = list(config.keys())
values = [config[k] for k in keys]

for combo in product(*values):
    combo_dict = dict(zip(keys, combo))
    job_name = "_".join(f"{k.upper()}{v}" for k,v in combo_dict.items())
    args = " ".join(str(v) for v in combo_dict.values())
    print(f"qsub -N {job_name} {script} {args}")
END
)

# Soumettre tous les jobs
eval "$COMBOS"
