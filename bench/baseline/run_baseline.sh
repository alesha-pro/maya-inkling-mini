#!/usr/bin/env bash
# Línea base fija de maya-ik2 (paso 0). Ejecutar desde la raíz del repo, dentro del contenedor `maya`.
# Uso: bench/baseline/run_baseline.sh [repeticiones]   → bench/baseline/results/<fecha>/
set -u
REPS=${1:-3}
OUT=bench/baseline/results/$(date +%Y%m%d-%H%M%S)${TAG:+-$TAG}
mkdir -p "$OUT"
export IK2_KV=fp8 IK2_GATE_TILE=64 IK2_GATE_BF16=1 IK2_Q8_UNROLL=1 IK2_BATCH_PREFILL=1
M=models/Inkling-2-Mini-UD-Q4_K_XL.gguf
{ echo "commit: $(git rev-parse HEAD 2>/dev/null)"; nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,clocks.max.sm --format=csv,noheader
  nvcc --version | tail -1; sha256sum build/maya-ik2; env | grep -E "^(IK2_|CTX_)" | sort; } > "$OUT/env.txt"
run() { # nombre, args...
  local name=$1; shift
  for r in $(seq 1 "$REPS"); do
    echo ">> $name rep $r"
    "$@" > "$OUT/$name.r$r.log" 2>&1
    echo "   exit=$? ; $(grep -E -i 'tok/s' "$OUT/$name.r$r.log" | tail -3 | tr '\n' ' ')"
  done
}
for n in 512 4096 16384; do
  run plain-$n build/maya-ik2 --model $M --ctx ${CTX_PLAIN:-344064} --chunk 2048 --tokens @bench/baseline/prompts/doc-$n.txt --gen 400
done
for n in 512 4096; do
  IK2_DRAFT_CTX=65536 run draft-$n build/maya-ik2 --model $M --ctx ${CTX_DRAFT:-262144} --chunk 2048 \
    --tokens @bench/baseline/prompts/doc-$n.txt --gen 384 --draft models/mtp.safetensors
done
echo "resultados en $OUT"
