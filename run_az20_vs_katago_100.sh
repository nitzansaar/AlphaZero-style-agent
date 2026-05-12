#!/usr/bin/env bash
# Run a 100-game alternating-colour 19x19 match between AZ20 and pretrained KataGo.
#
# Defaults are chosen for a handicapped KataGo ladder:
#   KATAGO_ELO=1 KATAGO_VISITS=1
#
# Override knobs at launch, for example:
#   AZ_ITERATION=146 KATAGO_ELO=482 KATAGO_VISITS=10 AZ_BATCH=16 ./run_az20_vs_katago_100.sh

set -euo pipefail
cd "$(dirname "$0")"

export BOARD_SIZE=19
export PYTHONUNBUFFERED=1

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

AZ_MODEL_DIR="${AZ_MODEL_DIR:-models_19x19_az20}"
AZ_ITERATION="${AZ_ITERATION:-latest}"
GAMES="${GAMES:-100}"
AZ_SIMS="${AZ_SIMS:-160}"
AZ_BATCH="${AZ_BATCH:-32}"
KATAGO_ELO="${KATAGO_ELO:-482}"
KATAGO_VISITS="${KATAGO_VISITS:-60}"
VERBOSE="${VERBOSE:-0}"

latest_iteration() {
    "$PYTHON_BIN" - <<'PY'
from pathlib import Path

base = Path("models_19x19_az20")
iters = []
if base.is_dir():
    for child in base.iterdir():
        if child.is_dir() and child.name.isdigit():
            if (child / "model_ts.pt").is_file() or (child / "model.pt").is_file():
                iters.append(int(child.name))
if not iters:
    raise SystemExit("No AlphaZero checkpoints found in models_19x19_az20")
print(max(iters))
PY
}

if [[ "$AZ_ITERATION" == "latest" ]]; then
    AZ_ITERATION="$(latest_iteration)"
fi

if [[ -n "${AZ_MODEL:-}" ]]; then
    AZ_MODEL_PATH="$AZ_MODEL"
else
    AZ_MODEL_PATH="${AZ_MODEL_DIR}/${AZ_ITERATION}/model_ts.pt"
fi

if [[ ! -f "$AZ_MODEL_PATH" ]]; then
    STATE_DICT_PATH="${AZ_MODEL_DIR}/${AZ_ITERATION}/model.pt"
    if [[ -f "$STATE_DICT_PATH" ]]; then
        echo "Exporting TorchScript model for AZ iteration ${AZ_ITERATION}..."
        "$PYTHON_BIN" export_model.py "$STATE_DICT_PATH" "$AZ_MODEL_PATH"
    else
        echo "ERROR: AlphaZero model not found: $AZ_MODEL_PATH" >&2
        echo "       Also missing state dict: $STATE_DICT_PATH" >&2
        exit 1
    fi
fi

KATAGO_MODEL="pretrained_katago_models/katago-elo-${KATAGO_ELO}.gz"
if [[ ! -f "$KATAGO_MODEL" ]]; then
    echo "ERROR: KataGo ELO model not found: $KATAGO_MODEL" >&2
    echo "Available exact ELOs:" >&2
    "$PYTHON_BIN" - <<'PY' >&2
from pathlib import Path
import re

base = Path("pretrained_katago_models")
elos = []
for path in base.glob("katago-elo-*.gz"):
    match = re.fullmatch(r"katago-elo-(\d+)\.gz", path.name)
    if match:
        elos.append(int(match.group(1)))
print(" ".join(map(str, sorted(elos))))
PY
    exit 1
fi

mkdir -p results
STAMP="$(date '+%Y%m%d-%H%M%S')"
CSV_PATH="${CSV_PATH:-results/az20_iter$(printf '%04d' "$AZ_ITERATION")_vs_katago_elo${KATAGO_ELO}_v${KATAGO_VISITS}_${GAMES}g_${STAMP}.csv}"

cmd=(
    "$PYTHON_BIN" az_vs_katago_19x19.py
    --games "$GAMES"
    --az-model "$AZ_MODEL_PATH"
    --az-sims "$AZ_SIMS"
    --az-batch "$AZ_BATCH"
    --katago-elo "$KATAGO_ELO"
    --katago-visits "$KATAGO_VISITS"
    --az-first-color alternate
    --csv "$CSV_PATH"
)

if [[ "$VERBOSE" == "1" || "$VERBOSE" == "true" ]]; then
    cmd+=(--verbose)
fi

echo "AlphaZero iteration : $AZ_ITERATION"
echo "AlphaZero model     : $AZ_MODEL_PATH"
echo "AlphaZero sims/batch: $AZ_SIMS / $AZ_BATCH"
echo "KataGo ELO/visits   : $KATAGO_ELO / $KATAGO_VISITS"
echo "Games               : $GAMES, alternating colours"
echo "CSV                 : $CSV_PATH"
printf '$'
printf ' %q' "${cmd[@]}"
printf '\n\n'

"${cmd[@]}"

CSV_PATH="$CSV_PATH" AZ_ITERATION="$AZ_ITERATION" KATAGO_ELO="$KATAGO_ELO" KATAGO_VISITS="$KATAGO_VISITS" "$PYTHON_BIN" - <<'PY'
import csv
import math
import os
from pathlib import Path

csv_path = Path(os.environ["CSV_PATH"])
rows = list(csv.DictReader(csv_path.open(newline="")))
if not rows:
    raise SystemExit(f"No game rows found in {csv_path}")

total = len(rows)
draws = sum(row["winner"] == "Draw" for row in rows)
az_wins = sum(row["az_won"] == "1" for row in rows)
kg_wins = total - az_wins - draws
score = az_wins + 0.5 * draws
score_rate = score / total
se = math.sqrt(max(score_rate * (1.0 - score_rate), 0.0) / total)
ci_low = max(0.0, score_rate - 1.96 * se)
ci_high = min(1.0, score_rate + 1.96 * se)

az_black = [row for row in rows if row["az_color"] == "Black"]
az_white = [row for row in rows if row["az_color"] == "White"]
az_black_wins = sum(row["az_won"] == "1" for row in az_black)
az_white_wins = sum(row["az_won"] == "1" for row in az_white)
avg_moves = sum(int(row["num_moves"]) for row in rows) / total

print()
print("Summary")
print("=======")
print(f"AZ iteration {os.environ['AZ_ITERATION']} vs KataGo ELO {os.environ['KATAGO_ELO']} visits {os.environ['KATAGO_VISITS']}")
print(f"Games        : {total}")
print(f"AZ wins      : {az_wins}")
print(f"KataGo wins  : {kg_wins}")
print(f"Draws        : {draws}")
print(f"AZ score     : {score:.1f}/{total} = {score_rate:.1%}  (approx 95% CI {ci_low:.1%}-{ci_high:.1%})")
print(f"AZ as Black  : {az_black_wins}/{len(az_black)}")
print(f"AZ as White  : {az_white_wins}/{len(az_white)}")
print(f"Avg moves    : {avg_moves:.1f}")
print(f"CSV          : {csv_path}")
PY
