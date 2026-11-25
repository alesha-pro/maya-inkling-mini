#!/usr/bin/env bash
# Control de paridad del motor: maya-ik2 (GPU) sobre cada prompt contra bench/parity/ref/*.npz, en tres modos:
#   dump    --dump: el camino general, las 512 posiciones (y las capas, si está la referencia completa)
#   decode  prellenado de K tokens (supercapas) y el resto de uno en uno por el camino residente (fast_token + grafo
#           CUDA), forzados con IK2_FORCE; logits de IK2_LOGITS_OUT: lo que corre el servidor al decodificar
#   block   igual, pero en ventanas de W por el camino de bloques (IK2_FORCE_BLOCK): la verificación del drafter
# Cada modo da su veredicto (tools/ik2_compare.py --gate) y su resumen en bench/parity/results/.
# Sale con 1 si algún modo falla un umbral, 2 si faltan datos o el motor falla. Necesita la GPU libre (~16 GiB).
# Uso: [MODES="dump decode block"] [K=128] [W=5] bench/parity/run_gate.sh [--flags-de-ik2_compare]
set -u
PY=${PY:-python3}
M=${MODEL:-models/Inkling-2-Mini-UD-Q4_K_XL.gguf}
MODES=${MODES:-dump decode block}
K=${K:-128}                                         # decode/block: tokens de prellenado (> 64: supercapas)
W=${W:-5}                                           # block: ventana (5 = el ancho por defecto del drafter)
( cd bench/parity && sha256sum -c --quiet MANIFEST.sha256 ) || { echo "MANIFEST no coincide: prompts o referencias cambiaron"; exit 2; }
export IK2_KV=${IK2_KV:-f16}                        # paridad con caché FP16: FP8 deriva entre ejecuciones (doc)
sha=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p bench/parity/results
worst=0
for mode in $MODES; do
  pairs=()
  for p in bench/parity/prompts/*.txt; do
    n=$(basename "$p" .txt); d=data/ik2/parity/eng-$mode-$n
    [ "$mode" = dump ] && d=data/ik2/parity/eng-$n             # (el nombre de la primera versión)
    rm -rf "$d"; mkdir -p "$d"
    case $mode in
      dump)
        build/maya-ik2 --model "$M" --ctx 8192 --chunk 2048 --tokens @"$p" --gen 0 --dump "$d" > "$d/run.log" 2>&1 ;;
      decode|block)
        tr ',' '\n' < "$p" | grep . > "$d/all.txt"
        N=$(wc -l < "$d/all.txt")
        head -n "$K" "$d/all.txt" | paste -sd, > "$d/prompt.txt"
        tail -n +"$((K + 1))" "$d/all.txt" | paste -sd, > "$d/force.txt"
        echo "$K" > "$d/first_pos"
        extra=(); [ "$mode" = block ] && extra=(env IK2_FORCE_BLOCK="$W")
        "${extra[@]}" env IK2_FORCE="$d/force.txt" IK2_LOGITS_OUT="$d/logits.f32" \
          build/maya-ik2 --model "$M" --ctx 8192 --chunk 2048 --tokens @"$d/prompt.txt" --gen $((N - K + 1)) \
          > "$d/run.log" 2>&1 ;;
      *) echo "modo desconocido: $mode"; exit 2 ;;
    esac
    [ $? -eq 0 ] || { echo "$mode/$n: el motor falló (ver $d/run.log)"; exit 2; }
    if [ "$mode" != dump ]; then
      grep -q "tokens take the resident path" "$d/run.log" || { echo "$mode/$n: el motor no quedó residente (ver $d/run.log)"; exit 2; }
      [ "$mode" = decode ] || grep -q "forced block:" "$d/run.log" || { echo "$mode/$n: no pasó por el camino de bloques"; exit 2; }
    fi
    ref=bench/parity/ref/$n.npz; [ -f data/ik2/parity/ref-$n/logprobs.npy ] && ref=data/ik2/parity/ref-$n   # completo: KL exacto + capas
    pairs+=("$d" "$ref")
  done
  out=bench/parity/results/$stamp-$sha-$mode.json
  echo "== modo $mode"
  "$PY" tools/ik2_compare.py "${pairs[@]}" --gate --json "$out" "$@"
  rc=$?; echo "resumen: $out"
  [ $rc -gt $worst ] && worst=$rc
done
exit $worst
