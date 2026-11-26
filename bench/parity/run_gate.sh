#!/usr/bin/env bash
# Control de paridad del motor: maya-ik2 (GPU) sobre cada prompt contra bench/parity/ref/*.npz, en tres modos:
#   dump    --dump: el camino general, las 512 posiciones (y las capas, si está la referencia completa)
#   decode  prellenado de K tokens (supercapas) y el resto de uno en uno por el camino residente (fast_token + grafo
#           CUDA), forzados con IK2_FORCE; logits de IK2_LOGITS_OUT: lo que corre el servidor al decodificar
#   block   igual, pero en ventanas de W por el camino de bloques (IK2_FORCE_BLOCK): la verificación del drafter
#   cpu8    igual que decode, pero con la config de 8 GB (docs/inkling2-mini/ik2-mini-8gb.json): un caché de experts de
#           CACHE_GB (3.8: el de la RTX 5070 Laptop) y la CPU calculando los que faltan (IK2_CPU, IK2_CPU_Q8, densos
#           Q8_0). En una GPU grande emula la chica; el motor tiene que no quedar residente y usar la CPU
# Cada modo da su veredicto (tools/ik2_compare.py --gate) y su resumen en bench/parity/results/.
# Sale con 1 si algún modo falla un umbral, 2 si faltan datos o el motor falla. Necesita la GPU libre (~16 GiB).
# Uso: [MODES="dump decode block"] [K=128] [W=5] [CACHE_GB=3.8] [BIN=build/maya-ik2] bench/parity/run_gate.sh [--flags]
# (cpu8 no está en los modos por defecto: necesita ~15 GiB de RAM fijada; MODES="dump decode block cpu8")
set -u
PY=${PY:-python3}
M=${MODEL:-models/Inkling-2-Mini-UD-Q4_K_XL.gguf}
MODES=${MODES:-dump decode block}
K=${K:-128}                                         # decode/block: tokens de prellenado (> 64: supercapas)
W=${W:-5}                                           # block: ventana (5 = el ancho por defecto del drafter)
BIN=${BIN:-build/maya-ik2}
CACHE_GB=${CACHE_GB:-3.8}                           # cpu8: el caché de experts en VRAM
# cpu8: el entorno de docs/inkling2-mini/ik2-mini-8gb.json (sin IK2_ROOM_MIB: aquí la VRAM sobra)
CPU8_ENV=(IK2_CPU=1 IK2_UP_MS=5 IK2_CPU_Q8=1 IK2_LAZY_CPU=1 IK2_LAZY_MAX=2 IK2_LAZY_PER=1 IK2_DENSE=q8
          IK2_GATE_TILE=64 IK2_GATE_BF16=1 IK2_Q8_UNROLL=1 IK2_BATCH_PREFILL=1)
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
        "$BIN" --model "$M" --ctx 8192 --chunk 2048 --tokens @"$p" --gen 0 --dump "$d" > "$d/run.log" 2>&1 ;;
      decode|block|cpu8)
        tr ',' '\n' < "$p" | grep . > "$d/all.txt"
        N=$(wc -l < "$d/all.txt")
        head -n "$K" "$d/all.txt" | paste -sd, > "$d/prompt.txt"
        tail -n +"$((K + 1))" "$d/all.txt" | paste -sd, > "$d/force.txt"
        echo "$K" > "$d/first_pos"
        extra=(); args=(--ctx 8192 --chunk 2048)
        [ "$mode" = block ] && extra=(IK2_FORCE_BLOCK="$W")
        [ "$mode" = cpu8 ] && extra=("${CPU8_ENV[@]}") && args=(--ctx 8192 --chunk 512 --cache-gb "$CACHE_GB" --threads 8)
        env "${extra[@]}" IK2_FORCE="$d/force.txt" IK2_LOGITS_OUT="$d/logits.f32" \
          "$BIN" --model "$M" "${args[@]}" --tokens @"$d/prompt.txt" --gen $((N - K + 1)) \
          > "$d/run.log" 2>&1 ;;
      *) echo "modo desconocido: $mode"; exit 2 ;;
    esac
    [ $? -eq 0 ] || { echo "$mode/$n: el motor falló (ver $d/run.log)"; exit 2; }
    if [ "$mode" = cpu8 ]; then
      ! grep -q "tokens take the resident path" "$d/run.log" || { echo "$mode/$n: quedó residente: no emula 8 GB"; exit 2; }
      # (la vía de CPU necesita AVX-512: sin él el motor la apaga y copia cada miss; REQUIRE_CPU=1 lo hace un fallo)
      cpu=$(grep -oE "CPU lane: [0-9]+ threads, [0-9]+ experts" "$d/run.log" | grep -oE "[0-9]+ experts" | grep -oE "[0-9]+")
      if [ "${cpu:-0}" -gt 0 ]; then echo "$mode/$n: $cpu experts en la CPU"
      elif [ "${REQUIRE_CPU:-0}" = 1 ]; then echo "$mode/$n: la CPU no calculó ningún expert (ver $d/run.log)"; exit 2
      else echo "$mode/$n: AVISO: sin vía de CPU (¿sin AVX-512?): solo caché + copias"; fi
    elif [ "$mode" != dump ]; then
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
