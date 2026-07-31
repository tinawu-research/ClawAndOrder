#!/usr/bin/env bash
# Ship the corpus to the fine-tuning node and launch the LoRA run in tmux.
#
# Two nodes, different accounts: the agent/brain node runs as cognitivo_g17,
# the fine-tuning node as cognitivo_g18. Everything below runs remotely.
#
# tmux is not optional. The handout reports earlyoom killing detached training
# runs, and nothing is checkpointed before step 20 (~25-35 min in), so a killed
# run before then is a full restart.
#
#   bash training/scripts/train.sh              # ship + launch
#   bash training/scripts/train.sh --dry-run    # ship + validate config only
#
set -euo pipefail

FT_HOST="${FT_HOST:-cognitivo_g18@10.0.1.11}"
FT_WORK="${FT_WORK:-/home/cognitivo_g18/finetune}"
NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:25.09}"
MODEL_DIR="${MODEL_DIR:-/home/cognitivo_g18/local-llm-setup/models}"
SESSION="${SESSION:-nemotron-lora}"
CONFIG_NAME="lora_nemotron_r32.yaml"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="1"

echo "==> checking corpus"
for f in training/data/train.io.jsonl training/data/val.io.jsonl; do
  [[ -s "$REPO_ROOT/$f" ]] || { echo "missing or empty: $f (run build.py)"; exit 1; }
done
wc -l "$REPO_ROOT"/training/data/{train,val}.io.jsonl

echo "==> shipping to $FT_HOST:$FT_WORK"
ssh "$FT_HOST" "mkdir -p '$FT_WORK/training/data' '$FT_WORK/training/configs' '$FT_WORK/checkpoints' '$FT_WORK/logs'"
rsync -az --info=stats1 \
  "$REPO_ROOT/training/data/train.io.jsonl" \
  "$REPO_ROOT/training/data/val.io.jsonl" \
  "$FT_HOST:$FT_WORK/training/data/"
rsync -az \
  "$REPO_ROOT/training/configs/$CONFIG_NAME" \
  "$FT_HOST:$FT_WORK/training/configs/"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$FT_WORK/logs/nemo_lora_${STAMP}.log"

# --ipc=host plus the two ulimits are what the NeMo container asks for; the
# default 64MB shmem is not enough for its dataloader workers.
#
# Launched with plain python, deliberately not torchrun.
#
# AutoModel's single-worker path (components/distributed/init_utils.py) picks its
# own free port and calls init_process_group with
# init_method="tcp://localhost:<port>". Under torchrun, TORCHELASTIC_USE_AGENT_STORE=True
# puts the TCPStore into *client* mode, so rank 0 tries to connect to that port
# instead of listening on it, and the run hangs in rendezvous until the
# distributed timeout expires. Verified directly in this container:
#
#     under torchrun:  AGENT_STORE=True  -> FAIL (client socket timed out)
#     plain python:    AGENT_STORE=None  -> OK
#
# A single-process, single-GPU run needs nothing torchrun provides.
DOCKER_CMD=$(cat <<EOF
docker run --rm --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $FT_WORK:/workspace \
  -v $MODEL_DIR:/workspace/models:ro \
  -w /opt/Automodel \
  $NEMO_IMAGE \
  python examples/llm_finetune/finetune.py \
    --config /workspace/training/configs/$CONFIG_NAME
EOF
)

if [[ -n "$DRY_RUN" ]]; then
  echo "==> dry run: validating config loads and data is readable"
  ssh "$FT_HOST" "docker run --rm -v $FT_WORK:/workspace $NEMO_IMAGE \
    python -c \"
import yaml, json, sys
cfg = yaml.safe_load(open('/workspace/training/configs/$CONFIG_NAME'))
print('recipe keys:', sorted(cfg))
print('max_steps  :', cfg['step_scheduler']['max_steps'])
print('lora dim   :', cfg['peft']['dim'])
print('seq_length :', cfg['dataset']['seq_length'])
p = cfg['dataset']['path_or_dataset_id']
rows = [json.loads(l) for l in open(p)]
print('train rows :', len(rows))
print('columns    :', sorted(rows[0]))
\""
  echo "==> dry run OK"
  exit 0
fi

echo "==> launching in tmux session '$SESSION'"
ssh "$FT_HOST" "tmux kill-session -t $SESSION 2>/dev/null || true"
ssh "$FT_HOST" "tmux new-session -d -s $SESSION \"$DOCKER_CMD 2>&1 | tee $LOG\""

echo
echo "launched. log: $FT_HOST:$LOG"
echo
echo "  follow:   ssh $FT_HOST 'tail -f $LOG'"
echo "  attach:   ssh -t $FT_HOST 'tmux attach -t $SESSION'"
echo "  stop:     ssh $FT_HOST 'tmux kill-session -t $SESSION'"
echo
echo "nothing checkpoints before step 20 (~25-35 min). Watch for a loss spike"
echo "around step 50 -- at lr 5e-5 there should not be one."
