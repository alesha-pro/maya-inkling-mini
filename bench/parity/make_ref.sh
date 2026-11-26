#!/usr/bin/env bash
# Referencia float64 (ref/inkling2.py, CPU) de cada prompt de bench/parity/prompts → bench/parity/ref/<nombre>.npz
# (top-256 por posición, cabe en el repo) + los seams completos en data/ik2/parity/ref-<nombre>/ (grandes, no se versionan).
# Una vez por cambio de modelo o de referencia, no por cambio del motor. ~15–30 min por prompt de 512 tokens.
# Uso: PY=~/.cache/maya-ref-venv/bin/python bench/parity/make_ref.sh [nombre ...]
set -eu
PY=${PY:-python3}; THREADS=${THREADS:-6}
M=${MODEL:-models/Inkling-2-Mini-UD-Q4_K_XL.gguf}
names=("$@"); [ ${#names[@]} -gt 0 ] || names=($(ls bench/parity/prompts | sed 's/\.txt$//'))
for n in "${names[@]}"; do
  out=data/ik2/parity/ref-$n
  [ -f bench/parity/ref/$n.npz ] && { echo "$n: ya existe, se salta"; continue; }
  mkdir -p "$out"; s=$(date +%s)
  nice -n 19 env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    "$PY" ref/inkling2.py "$M" --tokens "$(cat bench/parity/prompts/$n.txt)" --dump-dir "$out" > "$out/run.log" 2>&1
  "$PY" tools/ik2_compare.py --compact "$out" bench/parity/ref/$n.npz --topk 256
  echo "$n: $(( $(date +%s) - s )) s"
done
( cd bench/parity && sha256sum prompts/*.txt ref/*.npz > MANIFEST.sha256 )
echo "modelo: $(stat -c '%n %s bytes' "$M")" > bench/parity/MODEL.txt
