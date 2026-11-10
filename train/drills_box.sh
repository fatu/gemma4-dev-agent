#!/usr/bin/env bash
# Step 2 of the E4B gap plan on the GPU box (BOX_PLAN.md B-3p): an edit-skill LoRA from checked edit_file drills
# (train/edit_drills.py) mixed with E4B's own resolved calls on SWE-rebench tasks, gated offline on held-out drills
# before any agent eval. One stage per call; no `set -e`: every stage prints its result and ends with a 📷 line.
#   bash train/drills_box.sh build          # CPU: training + held-out drills + the held-out block and retry slices (≈ 20 min)
#   GPU=0 bash train/drills_box.sh gate0    # E4B, no adapter, on the held-out drills (temperature 0) → the baseline gate
#   GPU=0 bash train/drills_box.sh think    # E4B's own thoughts into the training drills (temperature 0.7, THINK_SAMPLES
#                                           # samples; a thought is kept only from a sample whose own call was the answer)
#   bash train/drills_box.sh mix            # render + mix with runs/sft/rebenche4b (drills ≤ DRILL_SHARE of label tokens)
#   GPU=0 bash train/drills_box.sh train    # the LoRA (one card)
#   GPU=0 bash train/drills_box.sh gate1    # each epoch's adapter on the held-out drills, paired against gate0's per-drill
#                                           # outcomes (McNemar), verdict by tier (edit_drills.py TIER_AB / TIER_STRETCH)
#   bash train/drills_box.sh subs           # the chosen epoch → runs/adapters/$NAME + a build check
# gate0 / think / gate1 start their own server on $GPU and stop every `vllm serve` first: run them when no lane runs —
# or, beside running lanes (e.g. E4B's rollouts on GPUs 1-3), with KILL_ALL=0 DPORT=8010: then only this script's own
# server (vllm serve … --port $DPORT) is started and stopped, and the lanes keep theirs. Start the lanes FIRST
# (rebench_gen.sh gen stops every vllm serve when it starts).
# Knobs (every one printed): GPU · N_TRAIN (2800) · TRAIN_FAMILIES (the training set's family weights: more anchor / block
# / retry than the held-out sets, which keep edit_drills.py's FAMILIES) · FILE_CAP (25 drills per training file) · N_HELD (300) · N_BLOCK (80, the held-out block slice) · N_RETRY (80, the
# held-out retry slice: E4B's failed-edit loops, read with its `resend` line) · TRAIN_LIBS ·
# HELD_LIBS · BLOCK_LIBS / HELD_BLOCK_LIBS (the block family's repetitive libraries) · SYSTEM (the arm's system.md)
# · THINK_SAMPLES (4) · DROP_TOOLS (for a tools6 arm) · TOOLS_FROM (a traj log; default the first runs/*/traj/*.jsonl) · OWN (rebenche4b
# rendered file) · DRILL_SHARE (0.6: the drills keep at most this share of the mix's label tokens; 0.4 until B-3q) · TRAIN_PY (.venv-train/bin/python) · DQ (bf16 base) · EPOCHS LR ACCUM R ALPHA · EVAL_EVERY (100 steps; 25 until B-3q: the held-out eval then cost as much as training) ·
# NAME (e4b_drill) · EPOCH (subs: default the gate1 winner, else the lower held-out NLL).
# LoRA defaults r16 α16 (scale 1, lr the only strength knob; 2026-11-08, from the Kaggle survey in the private notes:
# α/r 2 looped, rank 4–64 made no difference, a smaller rank leaves the scorer more KV cache). Train two NAMEs with
# LR=1e-5 and LR=3e-5 on two cards (BOX_PLAN B-3p) and let gate1 pick.
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?build|gate0|think|mix|train|gate1|subs}"; T0=$(date +%s)
say() { echo "== $(date '+%T') [$(( $(date +%s) - T0 ))s] $*"; }
GPU="${GPU:-0}"; KILL_ALL="${KILL_ALL:-1}"; DPORT="${DPORT:-8000}"; PY=.venv-eval/bin/python; TRAIN_PY="${TRAIN_PY:-.venv-train/bin/python}"
N_TRAIN="${N_TRAIN:-2800}"; N_HELD="${N_HELD:-300}"; FILE_CAP="${FILE_CAP:-25}"
# B-3q (2026-11-09): only 478 of 1,500 drills at FAMILIES' weights exercised "a unique anchor while a repeat is in view"
# (E4B's several-matches failure); ~1,500 such drills are the estimate for several matches ≤ 10 %
TRAIN_FAMILIES="${TRAIN_FAMILIES:-copy=0.27,anchor=0.18,recover=0.18,stale=0.07,block=0.15,retry=0.15}"
TRAIN_LIBS="${TRAIN_LIBS:-click,jinja2,attr,anyio,h11,werkzeug,packaging,markdown_it,yaml,dateutil}"
HELD_LIBS="${HELD_LIBS:-sqlalchemy,tqdm,tenacity,starlette}"; N_BLOCK="${N_BLOCK:-80}"; N_RETRY="${N_RETRY:-80}"
BLOCK_LIBS="${BLOCK_LIBS:-typer,openai,networkx}"; HELD_BLOCK_LIBS="${HELD_BLOCK_LIBS:-sqlmodel,huggingface_hub,pandas}"
SYSTEM="${SYSTEM:-scripts/arms/latest_e4b/prompts/system.md}"; DROP_TOOLS="${DROP_TOOLS:-}"
TOOLS_FROM="${TOOLS_FROM:-$(ls runs/*/traj/*.jsonl runs/*/*/traj/*.jsonl 2>/dev/null | grep -v '\.tools\.jsonl$' | head -1)}"
OWN="${OWN:-runs/sft/rebenche4b.rendered.jsonl}"; DRILL_SHARE="${DRILL_SHARE:-0.6}"; THINK_SAMPLES="${THINK_SAMPLES:-4}"
DQ="${DQ:-runs/dequant/gemma-4-E4B-dq}"; NAME="${NAME:-e4b_drill}"; OUTD="runs/sft_out/$NAME"
E4B=google/gemma-4-E4B-it-qat-w4a16-ct; ALIAS=gemma-4-e4b-it-qat-w4a16-ct; TOK="${TOK:-$E4B}"
D=runs/sft/drills; H=runs/sft/drills_heldout; HB=runs/sft/drills_heldout_block; HR=runs/sft/drills_heldout_retry; G=runs/drills
SFT="--base $DQ --max-len 32768 --epochs ${EPOCHS:-2} --lr ${LR:-3e-5} --warmup 0.05 --accum ${ACCUM:-16} --r ${R:-16} --alpha ${ALPHA:-16} --clip 1.0 --eval-every ${EVAL_EVERY:-100} --seed 0"
if [ "$KILL_ALL" = 0 ] && [ "$DPORT" = 8000 ]; then echo "KILL_ALL=0 needs DPORT ≠ 8000 (the lanes serve on 8000)"; exit 1; fi
echo "drills_box $STEP · GPU $GPU · server port $DPORT · stops $([ "$KILL_ALL" = 1 ] && echo 'every vllm serve' || echo "only its own (port $DPORT)") · train $N_TRAIN from $TRAIN_LIBS (block: $BLOCK_LIBS; families $TRAIN_FAMILIES; ≤ $FILE_CAP per file) · held out $N_HELD from $HELD_LIBS (block: $HELD_BLOCK_LIBS) + block slice $N_BLOCK + retry slice $N_RETRY · system $SYSTEM · drop tools [${DROP_TOOLS}] · tools from $TOOLS_FROM · own $OWN · drill share ≤ $DRILL_SHARE · think samples $THINK_SAMPLES · sft: $SFT · name $NAME"
mkdir -p "$G"

stop_serve() {   # KILL_ALL=1: every vllm serve on the box; 0: only ours (by its port)
  if [ "$KILL_ALL" = 1 ]; then pkill -f "[v]llm serve"; else pkill -f "[v]llm serve.*--port $DPORT"; fi
}
serve() {   # $1 = a dir with adapters/<name>/ to mount, or "" for none; stop_serve first
  stop_serve; sleep 5
  SUBMISSION="${1:-harness/sample_submission}" MODEL=$E4B TP=1 GPU_UTIL=0.85 CUDA_VISIBLE_DEVICES=$GPU PORT=$DPORT \
    nohup bash scripts/serve_vllm.sh > $G/vllm.log 2>&1 &
  for i in $(seq 1 180); do curl -s --max-time 2 http://127.0.0.1:$DPORT/v1/models >/dev/null 2>&1 && break; sleep 5; done
  echo "  server up after ~$((i * 5)) s: $(curl -s http://127.0.0.1:$DPORT/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null || echo 'NOT UP — tail '$G/vllm.log)"
}
URL="http://127.0.0.1:$DPORT/v1"

case "$STEP" in
build)
  [ -f "$TOOLS_FROM" ] || { echo "TOOLS_FROM=$TOOLS_FROM: no traj log"; exit 1; }
  # a new training set: an earlier build's thoughts, mix and gate0 no longer belong to it
  rm -rf ${D}_native ${D}_native.rendered.jsonl ${D}_native.render.log runs/sft/drills_mix.rendered.jsonl $G/gate0*.out.jsonl $G/gate0*.sum.json
  echo "  removed the earlier build's ${D}_native, drills_mix and gate0 outcomes (think, mix and gate0 run again on the new sets)"
  $PY train/edit_drills.py build --libs "$TRAIN_LIBS" --block-libs "$BLOCK_LIBS" --n "$N_TRAIN" --seed 0 --families "$TRAIN_FAMILIES" --file-cap "$FILE_CAP" --tools-from "$TOOLS_FROM" \
    --drop-tools "$DROP_TOOLS" --system "$SYSTEM" --out $D 2>&1 | grep -v LiteLLM | cut -c1-300
  $PY train/edit_drills.py build --libs "$HELD_LIBS" --block-libs "$HELD_BLOCK_LIBS" --n "$N_HELD" --seed 1 --tools-from "$TOOLS_FROM" \
    --drop-tools "$DROP_TOOLS" --system "$SYSTEM" --out $H 2>&1 | grep -v LiteLLM | cut -c1-300
  $PY train/edit_drills.py build --libs "${HELD_LIBS%%,*}" --block-libs "$HELD_BLOCK_LIBS" --families block=1 --n "$N_BLOCK" --seed 2 \
    --tools-from "$TOOLS_FROM" --drop-tools "$DROP_TOOLS" --system "$SYSTEM" --out $HB 2>&1 | grep -v LiteLLM | cut -c1-300
  $PY train/edit_drills.py build --libs "$HELD_LIBS" --block-libs "" --families retry=1 --n "$N_RETRY" --seed 4 --exclude $H \
    --tools-from "$TOOLS_FROM" --drop-tools "$DROP_TOOLS" --system "$SYSTEM" --out $HR 2>&1 | grep -v LiteLLM | cut -c1-300
  echo "📷 the four '== drills' lines and the tools / sampling line" ;;
gate0)
  serve ""
  $PY train/edit_drills.py decode $H --url $URL --model $ALIAS --temperature 0 --out $G/heldout_base_t0.jsonl 2>&1 | grep -E "^==|^edit_drills"
  $PY train/edit_drills.py score $H $G/heldout_base_t0.jsonl --outcomes $G/gate0.out.jsonl --summary $G/gate0.sum.json 2>&1 | grep -v LiteLLM | tee $G/gate0.txt
  $PY train/edit_drills.py decode $HB --url $URL --model $ALIAS --temperature 0 --out $G/heldblock_base_t0.jsonl 2>&1 | grep -E "^==|^edit_drills"
  echo "-- block slice"; $PY train/edit_drills.py score $HB $G/heldblock_base_t0.jsonl --outcomes $G/gate0_block.out.jsonl --summary $G/gate0_block.sum.json 2>&1 | grep -v LiteLLM | tee $G/gate0_block.txt | grep -E "edit outcomes|block slice"
  $PY train/edit_drills.py decode $HR --url $URL --model $ALIAS --temperature 0 --out $G/heldretry_base_t0.jsonl 2>&1 | grep -E "^==|^edit_drills"
  echo "-- retry slice"; $PY train/edit_drills.py score $HR $G/heldretry_base_t0.jsonl --outcomes $G/gate0_retry.out.jsonl --summary $G/gate0_retry.sum.json 2>&1 | grep -v LiteLLM | tee $G/gate0_retry.txt | grep -E "edit outcomes|resend|recover call 1"
  stop_serve
  echo "  per-drill outcomes kept for gate1's paired comparison: $G/gate0*.out.jsonl"
  echo "📷 everything from 'edit_drills score' down, the '-- block slice' and '-- retry slice' lines" ;;
think)
  serve ""
  $PY train/edit_drills.py decode $D --url $URL --model $ALIAS --temperature 0.7 --samples "$THINK_SAMPLES" --out $G/train_think_t07.jsonl 2>&1 | grep -E "^==|^edit_drills"
  $PY train/edit_drills.py think $D $G/train_think_t07.jsonl --out ${D}_native 2>&1 | grep -E "^==|^   (samples|calls with)"
  $PY train/edit_drills.py score $D $G/train_think_t07.jsonl 2>&1 | grep -E "edit outcomes|^== gate"   # E4B on its own training drills
  stop_serve
  echo "📷 the '== think' line, the 'samples:' and 'calls with a thought by family' lines and the two score lines" ;;
mix)
  [ -d ${D}_native ] || { echo "no ${D}_native: run think first (on this build)"; exit 1; }
  python3 -c "
import glob, os, sys
a = {os.path.basename(f) for f in glob.glob(sys.argv[1] + '/*.json')}; b = {os.path.basename(f) for f in glob.glob(sys.argv[2] + '/*.json')}
print(f'  think set vs build: {len(a & b)} of {len(a)} drills in both')
sys.exit(0 if a == b else 1)" $D ${D}_native || { echo "${D}_native is not this build's drills (an earlier think?): run think again"; exit 1; }
  for d in ${D}_native $H; do
    $PY train/render.py $d --tokenizer "$TOK" --pack --sample 100 --out $d.rendered.jsonl > $d.render.log 2>&1
    echo "  $d: $(grep -E '^calls' $d.render.log | cut -c1-120) · $(grep -E '^RESULT' $d.render.log)"
  done
  python3 - ${D}_native.rendered.jsonl "$OWN" "$DRILL_SHARE" runs/sft/drills_mix.rendered.jsonl ${D}_native <<'PY'
import glob, json, os, random, sys
from collections import Counter
drills, own, share, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
fam = {}
for f in glob.glob(sys.argv[5] + "/*.json"):
    r = json.load(open(f)); fam[r["instance_id"]] = r.get("family", "?")
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
n_all = Counter(fam.values()); n_kept = Counter(fam.get(i, "?") for i in {r["instance_id"] for r in d})
print("   drills kept by family (of built): " + " · ".join(f"{k} {n_kept[k]}/{n_all[k]}" for k in sorted(n_all))
      + ("" if ld <= ld2 else f"   ← the {share} cap dropped drills at random: raise DRILL_SHARE to keep them all"))
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
  [ -s $G/gate0.out.jsonl ] && [ -s $G/gate0_block.out.jsonl ] && [ -s $G/gate0_retry.out.jsonl ] || { echo "run gate0 first (on these held-out sets): gate1 is read against it, paired"; exit 1; }
  rm -rf runs/drill_serve; mkdir -p runs/drill_serve/adapters
  for e in $OUTD/epoch*/; do k=$(basename "$e"); [ -f "$e/adapter_model.safetensors" ] && cp -r "$e" "runs/drill_serve/adapters/${NAME}_$k"; done
  serve runs/drill_serve
  for a in $(ls runs/drill_serve/adapters); do
    $PY train/edit_drills.py decode $H --url $URL --model "$a" --temperature 0 --out $G/heldout_${a}_t0.jsonl 2>&1 | grep -E "^=="
    echo "-- $a"; $PY train/edit_drills.py score $H $G/heldout_${a}_t0.jsonl --baseline $G/gate0.out.jsonl --summary $G/gate1_$a.sum.json 2>&1 | grep -v LiteLLM | tee $G/gate1_$a.txt | grep -E "edit outcomes|^== gate|answer length"
    $PY train/edit_drills.py decode $HB --url $URL --model "$a" --temperature 0 --out $G/heldblock_${a}_t0.jsonl 2>&1 | grep -E "^=="
    $PY train/edit_drills.py score $HB $G/heldblock_${a}_t0.jsonl --baseline $G/gate0_block.out.jsonl --summary $G/gate1_block_$a.sum.json 2>&1 | grep -v LiteLLM | tee $G/gate1_block_$a.txt >/dev/null
    $PY train/edit_drills.py decode $HR --url $URL --model "$a" --temperature 0 --out $G/heldretry_${a}_t0.jsonl 2>&1 | grep -E "^=="
    $PY train/edit_drills.py score $HR $G/heldretry_${a}_t0.jsonl --baseline $G/gate0_retry.out.jsonl --summary $G/gate1_retry_$a.sum.json 2>&1 | grep -v LiteLLM | tee $G/gate1_retry_$a.txt >/dev/null
    $PY train/edit_drills.py tier --main $G/gate1_$a.sum.json --block $G/gate1_block_$a.sum.json --retry $G/gate1_retry_$a.sum.json --name "$a" 2>&1 | grep -v LiteLLM | tee $G/tier_$a.txt
  done
  echo "-- no adapter (gate0)"; grep -E "edit outcomes|^== gate|block slice|answer length|resend" $G/gate0.txt $G/gate0_block.txt $G/gate0_retry.txt 2>/dev/null || echo "  (no gate0 yet)"
  stop_serve
  echo "📷 each adapter's '-- <name>' block: edit outcomes, gate, answer length, and the whole 'edit_drills tier' block (✓/✗ lines + '== tier'); full scores in $G/gate1_*.txt" ;;
subs)
  BEST=$(python3 - "$G" "$NAME" <<'PY'
import glob, re, sys
best = None
for f in glob.glob(f"{sys.argv[1]}/gate1_{sys.argv[2]}_epoch*.txt"):          # only adapters whose tier says A/B tier PASS
    m = re.search(r"== gate: intended (\d+)%", open(f).read())
    t = f.replace("/gate1_", "/tier_")
    ok = "A/B tier PASS" in (open(t).read() if __import__("os").path.exists(t) else "")
    if m and ok and (best is None or int(m.group(1)) > best[0]):
        best = (int(m.group(1)), re.search(r"epoch(\d+)", f).group(1))
print(best[1] if best else "")
PY
)
  EP="${EPOCH:-$BEST}"
  say "subs: gate1 winner (A/B tier PASS, highest hit) epoch ${BEST:-none} → epoch ${EP:-?} (EPOCH=${EPOCH:-auto}) → runs/adapters/$NAME"
  [ -n "$EP" ] || { echo "no epoch of $NAME passed the A/B tier (runs/drills/tier_${NAME}_epoch*.txt): stop here, or name one with EPOCH=<n>"; exit 1; }
  [ -f "$OUTD/epoch$EP/adapter_model.safetensors" ] || { echo "no $OUTD/epoch$EP/adapter_model.safetensors"; exit 1; }
  rm -rf "runs/adapters/$NAME"; mkdir -p runs/adapters; cp -r "$OUTD/epoch$EP" "runs/adapters/$NAME"
  python3 -c "import json,sys;c=json.load(open(sys.argv[1]));print('  adapter_config: r',c.get('r'),'alpha',c.get('lora_alpha'),'base',c.get('base_model_name_or_path'))" "runs/adapters/$NAME/adapter_config.json"
  BUILD_ONLY=1 LATEST=e4b ADAPTER=$NAME SUBMISSION=runs/base_lb024 ARMS=base TAG=drillchk bash scripts/eval_dev.sh 2>&1 \
    | grep -E "latest set|latest e4b|^==" | cut -c1-220
  echo "📷 these lines" ;;
*) echo "unknown stage $STEP"; exit 1 ;;
esac
say "done ($STEP)"
