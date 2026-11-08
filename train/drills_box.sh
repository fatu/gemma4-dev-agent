#!/usr/bin/env bash
# Step 2 of the E4B gap plan on the GPU box (BOX_PLAN.md B-3p): an edit-skill LoRA from checked edit_file drills
# (train/edit_drills.py) mixed with E4B's own resolved calls on SWE-rebench tasks, gated offline on held-out drills
# before any agent eval. One stage per call; no `set -e`: every stage prints its result and ends with a 📷 line.
#   bash train/drills_box.sh build          # CPU: training + held-out drills + the held-out block slice (≈ 15 min)
#   GPU=0 bash train/drills_box.sh gate0    # E4B, no adapter, on the held-out drills (temperature 0) → the baseline gate
#   GPU=0 bash train/drills_box.sh think    # E4B's own thoughts into the training drills (temperature 0.7)
#   bash train/drills_box.sh mix            # render + mix with runs/sft/rebenche4b (drills ≤ DRILL_SHARE of label tokens)
#   GPU=0 bash train/drills_box.sh train    # the LoRA (one card)
#   GPU=0 bash train/drills_box.sh gate1    # each epoch's adapter on the held-out drills, beside gate0
#   bash train/drills_box.sh subs           # the chosen epoch → runs/adapters/$NAME + a build check
# gate0 / think / gate1 start their own server on $GPU and stop every `vllm serve` first: run them when no lane runs.
# Knobs (every one printed): GPU · N_TRAIN (1500) · N_HELD (300) · N_BLOCK (80, the held-out block slice) · TRAIN_LIBS ·
# HELD_LIBS · BLOCK_LIBS / HELD_BLOCK_LIBS (the block family's repetitive libraries) · SYSTEM (the arm's system.md)
# · DROP_TOOLS (for a tools6 arm) · TOOLS_FROM (a traj log; default the first runs/*/traj/*.jsonl) · OWN (rebenche4b
# rendered file) · DRILL_SHARE (0.4) · TRAIN_PY (.venv-train/bin/python) · DQ (bf16 base) · EPOCHS LR ACCUM R ALPHA ·
# NAME (e4b_drill) · EPOCH (subs: default the gate1 winner, else the lower held-out NLL).
# LoRA defaults r16 α16 (scale 1, lr the only strength knob; 2026-11-08, from the Kaggle survey in the private notes:
# α/r 2 looped, rank 4–64 made no difference, a smaller rank leaves the scorer more KV cache). Train two NAMEs with
# LR=1e-5 and LR=3e-5 on two cards (BOX_PLAN B-3p) and let gate1 pick.
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?build|gate0|think|mix|train|gate1|subs}"; T0=$(date +%s)
say() { echo "== $(date '+%T') [$(( $(date +%s) - T0 ))s] $*"; }
GPU="${GPU:-0}"; PY=.venv-eval/bin/python; TRAIN_PY="${TRAIN_PY:-.venv-train/bin/python}"
N_TRAIN="${N_TRAIN:-1500}"; N_HELD="${N_HELD:-300}"
TRAIN_LIBS="${TRAIN_LIBS:-click,jinja2,attr,anyio,h11,werkzeug,packaging,markdown_it,yaml,dateutil}"
HELD_LIBS="${HELD_LIBS:-sqlalchemy,tqdm,tenacity,starlette}"; N_BLOCK="${N_BLOCK:-80}"
BLOCK_LIBS="${BLOCK_LIBS:-typer,openai,networkx}"; HELD_BLOCK_LIBS="${HELD_BLOCK_LIBS:-sqlmodel,huggingface_hub,pandas}"
SYSTEM="${SYSTEM:-scripts/arms/latest_e4b/prompts/system.md}"; DROP_TOOLS="${DROP_TOOLS:-}"
TOOLS_FROM="${TOOLS_FROM:-$(ls runs/*/traj/*.jsonl runs/*/*/traj/*.jsonl 2>/dev/null | grep -v '\.tools\.jsonl$' | head -1)}"
OWN="${OWN:-runs/sft/rebenche4b.rendered.jsonl}"; DRILL_SHARE="${DRILL_SHARE:-0.4}"
DQ="${DQ:-runs/dequant/gemma-4-E4B-dq}"; NAME="${NAME:-e4b_drill}"; OUTD="runs/sft_out/$NAME"
E4B=google/gemma-4-E4B-it-qat-w4a16-ct; ALIAS=gemma-4-e4b-it-qat-w4a16-ct; TOK="${TOK:-$E4B}"
D=runs/sft/drills; H=runs/sft/drills_heldout; HB=runs/sft/drills_heldout_block; G=runs/drills
SFT="--base $DQ --max-len 32768 --epochs ${EPOCHS:-2} --lr ${LR:-3e-5} --warmup 0.05 --accum ${ACCUM:-16} --r ${R:-16} --alpha ${ALPHA:-16} --clip 1.0 --eval-every 25 --seed 0"
echo "drills_box $STEP · GPU $GPU · train $N_TRAIN from $TRAIN_LIBS (block: $BLOCK_LIBS) · held out $N_HELD from $HELD_LIBS (block: $HELD_BLOCK_LIBS) + block slice $N_BLOCK · system $SYSTEM · drop tools [${DROP_TOOLS}] · tools from $TOOLS_FROM · own $OWN · drill share ≤ $DRILL_SHARE · sft: $SFT · name $NAME"
mkdir -p "$G"

serve() {   # $1 = a dir with adapters/<name>/ to mount, or "" for none; stops every vllm serve on the box first
  pkill -f "[v]llm serve"; sleep 5
  SUBMISSION="${1:-harness/sample_submission}" MODEL=$E4B TP=1 GPU_UTIL=0.85 CUDA_VISIBLE_DEVICES=$GPU PORT=8000 \
    nohup bash scripts/serve_vllm.sh > $G/vllm.log 2>&1 &
  for i in $(seq 1 180); do curl -s --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 5; done
  echo "  server up after ~$((i * 5)) s: $(curl -s http://127.0.0.1:8000/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null || echo 'NOT UP — tail '$G/vllm.log)"
}

case "$STEP" in
build)
  [ -f "$TOOLS_FROM" ] || { echo "TOOLS_FROM=$TOOLS_FROM: no traj log"; exit 1; }
  $PY train/edit_drills.py build --libs "$TRAIN_LIBS" --block-libs "$BLOCK_LIBS" --n "$N_TRAIN" --seed 0 --tools-from "$TOOLS_FROM" \
    --drop-tools "$DROP_TOOLS" --system "$SYSTEM" --out $D 2>&1 | grep -v LiteLLM | cut -c1-300
  $PY train/edit_drills.py build --libs "$HELD_LIBS" --block-libs "$HELD_BLOCK_LIBS" --n "$N_HELD" --seed 1 --tools-from "$TOOLS_FROM" \
    --drop-tools "$DROP_TOOLS" --system "$SYSTEM" --out $H 2>&1 | grep -v LiteLLM | cut -c1-300
  $PY train/edit_drills.py build --libs "${HELD_LIBS%%,*}" --block-libs "$HELD_BLOCK_LIBS" --families block=1 --n "$N_BLOCK" --seed 2 \
    --tools-from "$TOOLS_FROM" --drop-tools "$DROP_TOOLS" --system "$SYSTEM" --out $HB 2>&1 | grep -v LiteLLM | cut -c1-300
  echo "📷 the three '== drills' lines and the tools / sampling line" ;;
gate0)
  serve ""
  $PY train/edit_drills.py decode $H --model $ALIAS --temperature 0 --out $G/heldout_base_t0.jsonl 2>&1 | grep -E "^==|^edit_drills"
  $PY train/edit_drills.py score $H $G/heldout_base_t0.jsonl 2>&1 | grep -v LiteLLM | tee $G/gate0.txt
  $PY train/edit_drills.py decode $HB --model $ALIAS --temperature 0 --out $G/heldblock_base_t0.jsonl 2>&1 | grep -E "^==|^edit_drills"
  echo "-- block slice"; $PY train/edit_drills.py score $HB $G/heldblock_base_t0.jsonl 2>&1 | grep -v LiteLLM | tee $G/gate0_block.txt | grep -E "edit outcomes|block slice"
  pkill -f "[v]llm serve"
  echo "📷 everything from 'edit_drills score' down (laptop expectation from the study: not_found ≈ 40 % at distance ≥ 2)" ;;
think)
  serve ""
  $PY train/edit_drills.py decode $D --model $ALIAS --temperature 0.7 --out $G/train_think_t07.jsonl 2>&1 | grep -E "^==|^edit_drills"
  $PY train/edit_drills.py think $D $G/train_think_t07.jsonl --out ${D}_native 2>&1 | grep -E "^=="
  $PY train/edit_drills.py score $D $G/train_think_t07.jsonl 2>&1 | grep -E "edit outcomes|^== gate"   # E4B on its own training drills
  pkill -f "[v]llm serve"
  echo "📷 the '== think' line and the two score lines" ;;
mix)
  for d in ${D}_native $H; do
    $PY train/render.py $d --tokenizer "$TOK" --pack --sample 100 --out $d.rendered.jsonl > $d.render.log 2>&1
    echo "  $d: $(grep -E '^calls' $d.render.log | cut -c1-120) · $(grep -E '^RESULT' $d.render.log)"
  done
  python3 - ${D}_native.rendered.jsonl "$OWN" "$DRILL_SHARE" runs/sft/drills_mix.rendered.jsonl <<'PY'
import json, os, random, sys
drills, own, share, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
lab = lambda r: sum(1 for x in r["labels"] if x != -100)
d = [json.loads(l) for l in open(drills)]
o = [json.loads(l) for l in open(own)] if os.path.exists(own) else []
if not o:
    print(f"  WARNING: no own calls ({own} missing) — drills only; run the E4B rollouts of B-3p first")
lo, ld = sum(map(lab, o)), sum(map(lab, d))
random.Random(0).shuffle(d)
if o and ld > share / (1 - share) * lo:                 # keep drills at most `share` of all label tokens
    cap, kept, acc = share / (1 - share) * lo, [], 0
    for r in d:
        if acc + lab(r) > cap: break
        kept.append(r); acc += lab(r)
    d = kept
mix = d + o
random.Random(1).shuffle(mix)
with open(out, "w") as w:
    for r in mix: w.write(json.dumps(r) + "\n")
ld2 = sum(map(lab, d))
print(f"== mix: drills {len(d)} sequences / {ld2} label tokens + own {len(o)} / {lo} → {len(mix)} sequences, "
      f"drill share {ld2 / max(1, ld2 + lo):.2f} (cap {share}) → {out}")
PY
  echo "📷 the two render lines and the '== mix' line" ;;
train)
  [ -s runs/sft/drills_mix.rendered.jsonl ] || { echo "run mix first"; exit 1; }
  [ -f "$DQ/config.json" ] || { echo "no bf16 base in $DQ: GPU=$GPU bash train/train_box.sh env dequant"; exit 1; }
  rm -rf "$OUTD"
  CUDA_VISIBLE_DEVICES=$GPU $TRAIN_PY train/sft_lora.py $SFT --data runs/sft/drills_mix.rendered.jsonl \
    --eval $H.rendered.jsonl --out "$OUTD" 2>&1 | grep -v Warning | tee runs/sft_$NAME.log | cut -c1-250
  echo "📷 the schedule line, the 'eval[...]' lines and both '== epoch' lines" ;;
gate1)
  rm -rf runs/drill_serve; mkdir -p runs/drill_serve/adapters
  for e in $OUTD/epoch*/; do k=$(basename "$e"); [ -f "$e/adapter_model.safetensors" ] && cp -r "$e" "runs/drill_serve/adapters/${NAME}_$k"; done
  serve runs/drill_serve
  for a in $(ls runs/drill_serve/adapters); do
    $PY train/edit_drills.py decode $H --model "$a" --temperature 0 --out $G/heldout_${a}_t0.jsonl 2>&1 | grep -E "^=="
    echo "-- $a"; $PY train/edit_drills.py score $H $G/heldout_${a}_t0.jsonl 2>&1 | grep -v LiteLLM | tee $G/gate1_$a.txt | grep -E "edit outcomes|distance [0-5]|^== gate|block slice"
    $PY train/edit_drills.py decode $HB --model "$a" --temperature 0 --out $G/heldblock_${a}_t0.jsonl 2>&1 | grep -E "^=="
    $PY train/edit_drills.py score $HB $G/heldblock_${a}_t0.jsonl 2>&1 | grep -v LiteLLM | tee $G/gate1_block_$a.txt | grep -E "block slice"
  done
  echo "-- no adapter (gate0)"; grep -E "edit outcomes|^== gate|block slice" $G/gate0.txt $G/gate0_block.txt 2>/dev/null || echo "  (no gate0 yet)"
  pkill -f "[v]llm serve"
  echo "📷 every '--' block (each adapter's outcomes, distance lines and gate) and the gate0 lines" ;;
subs)
  BEST=$(python3 - "$G" "$NAME" <<'PY'
import glob, re, sys
best = None
for f in glob.glob(f"{sys.argv[1]}/gate1_{sys.argv[2]}_epoch*.txt"):
    m = re.search(r"== gate: intended (\d+)%", open(f).read())
    if m and (best is None or int(m.group(1)) > best[0]):
        best = (int(m.group(1)), re.search(r"epoch(\d+)", f).group(1))
print(best[1] if best else "")
PY
)
  EP="${EPOCH:-$BEST}"
  say "subs: gate1 winner epoch ${BEST:-none} → epoch ${EP:-?} (EPOCH=${EPOCH:-auto}) → runs/adapters/$NAME"
  [ -f "$OUTD/epoch$EP/adapter_model.safetensors" ] || { echo "no $OUTD/epoch$EP/adapter_model.safetensors"; exit 1; }
  rm -rf "runs/adapters/$NAME"; mkdir -p runs/adapters; cp -r "$OUTD/epoch$EP" "runs/adapters/$NAME"
  python3 -c "import json,sys;c=json.load(open(sys.argv[1]));print('  adapter_config: r',c.get('r'),'alpha',c.get('lora_alpha'),'base',c.get('base_model_name_or_path'))" "runs/adapters/$NAME/adapter_config.json"
  BUILD_ONLY=1 LATEST=e4b ADAPTER=$NAME SUBMISSION=runs/base_lb024 ARMS=base TAG=drillchk bash scripts/eval_dev.sh 2>&1 \
    | grep -E "latest set|latest e4b|^==" | cut -c1-220
  echo "📷 these lines" ;;
*) echo "unknown stage $STEP"; exit 1 ;;
esac
say "done ($STEP)"
