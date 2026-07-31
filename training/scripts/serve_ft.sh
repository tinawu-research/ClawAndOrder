#!/usr/bin/env bash
# (Re)start the fine-tuned Nemotron server on the fine-tuning node.
#
# Serves the base weights plus every LoRA checkpoint as separate model ids, so
# the base-vs-fine-tuned sweep is one HTTP loop with a different `model` field
# instead of a restart per arm -- same process, same hardware state, comparable
# numbers.
#
#   bash training/scripts/serve_ft.sh                 # base only, raised context
#   bash training/scripts/serve_ft.sh --with-adapters # base + ck20..ck100
#
# Two settings that are easy to get wrong:
#   --max-lora-rank 32   vLLM defaults to 16 and will refuse a rank-32 adapter.
#   --max-model-len      raised from the original 4096; the planning loop
#                        overflowed that on multi-tool questions, which killed
#                        planning outright rather than merely truncating.
set -euo pipefail

FT_HOST="${FT_HOST:-cognitivo_g18@10.0.1.11}"
FT_WORK="${FT_WORK:-/home/cognitivo_g18/finetune}"
MODEL_DIR="${MODEL_DIR:-/home/cognitivo_g18/local-llm-setup/models}"
CKPT_DIR="${CKPT_DIR:-$FT_WORK/checkpoints/nemotron_synthesis_r32}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.45}"
NAME="nemotron-8b-finance"

WITH_ADAPTERS=""
[[ "${1:-}" == "--with-adapters" ]] && WITH_ADAPTERS="1"

LORA_ARGS=""
if [[ -n "$WITH_ADAPTERS" ]]; then
  # AutoModel writes the PEFT adapter to <ckpt>/epoch_0_step_<N>/model/, alongside
  # the tokenizer -- not to an hf_adapter/ directory. Locate by adapter_config.json
  # so this keeps working if the layout changes again.
  echo "==> locating adapters under $CKPT_DIR"
  MODULES=$(ssh "$FT_HOST" \
    "find '$CKPT_DIR' -name adapter_config.json -printf '%h\n' 2>/dev/null | sort -V" || true)
  [[ -n "$MODULES" ]] || { echo "no adapters found under $CKPT_DIR; train first"; exit 1; }

  MAPPING=""
  while read -r path; do
    [[ -z "$path" ]] && continue
    # .../epoch_0_step_19/model -> ck20. The directory name carries the 0-indexed
    # step, so step_19 is the checkpoint written *after* 20 optimizer steps.
    # Naming it ck20 keeps the arm labels aligned with ckpt_every_steps: 20.
    idx=$(echo "$path" | grep -oE 'step_[0-9]+' | grep -oE '[0-9]+' | tail -1)
    [[ -z "$idx" ]] && continue
    step=$((idx + 1))
    # find(1) ran on the host, but vLLM resolves these paths *inside* the
    # container, where $FT_WORK is mounted at /workspace. Passing the host path
    # gets LoRAAdapterNotFoundError and the server exits during startup.
    cpath="${path/#$FT_WORK//workspace}"
    MAPPING="$MAPPING ck${step}=${cpath}"
    echo "    ck${step} <- ${cpath}"
  done <<< "$MODULES"

  n_lora=$(wc -w <<< "$MAPPING")
  LORA_ARGS="--enable-lora --max-loras $n_lora --max-lora-rank 32 --lora-modules$MAPPING"
fi

echo "==> restarting vllm-ft (max_model_len=$MAX_LEN)"
ssh "$FT_HOST" "docker rm -f vllm-ft 2>/dev/null || true"
ssh "$FT_HOST" "docker run -d --name vllm-ft --gpus all --ipc=host \
  -p 8001:8000 \
  -v $MODEL_DIR:/models \
  -v $FT_WORK:/workspace \
  vllm/vllm-openai:latest \
  --model /models/Llama-3.1-Nemotron-Nano-8B-v1 \
  --served-model-name $NAME \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  $LORA_ARGS"

echo "==> waiting for readiness"
# Must check for actual JSON, not merely a reachable port: the port opens before
# the model finishes loading, and a bare curl exit code reports success while
# vLLM is still initialising -- or after it has died.
for i in $(seq 1 120); do
  if curl -s -m 3 "http://10.0.1.11:8001/v1/models" 2>/dev/null | grep -q '"data"'; then
    echo "    ready after ~$((i * 5))s"
    break
  fi
  if ! ssh "$FT_HOST" "docker ps -q --filter name=vllm-ft --filter status=running" | grep -q .; then
    echo "    container exited during startup:"
    ssh "$FT_HOST" "docker logs --tail 15 vllm-ft 2>&1" | sed 's/^/      /'
    exit 1
  fi
  sleep 5
done

echo
curl -s -m 10 http://10.0.1.11:8001/v1/models \
  | python3 -c "import sys,json;[print(' -',m['id'],'| len:',m.get('max_model_len')) for m in json.load(sys.stdin)['data']]" \
  || echo "server not answering yet; check: ssh $FT_HOST 'docker logs --tail 40 vllm-ft'"
