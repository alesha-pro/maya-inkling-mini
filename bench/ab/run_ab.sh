#!/usr/bin/env bash
# A/B de variables de entorno de maya-ik2 sobre la línea base (bench/baseline): las configuraciones se alternan en cada
# repetición (A B C A B C ...) para que la deriva térmica no favorezca a ninguna. Arranque en frío cada corrida.
# Uso (raíz del repo, GPU libre):
#   SCEN=decode|prefill REPS=3 bench/ab/run_ab.sh base= split66='IK2_EXPERT_MMA_SPLIT=6 IK2_EXPERT_MMA_SPLIT_DOWN=6' ...
#   decode:  prompt de 4096 tokens + 400 generados (mide la decodificación)
#   prefill: prompt de 16384 tokens + 8 generados (mide el prellenado)
# → bench/ab/results/<fecha>-<SCEN>/ con un log por corrida, gpu.txt y summary.md (mediana y rango por configuración).
# Ojo: la GPU comparte el escritorio y hay tramos lentos (−15..−25 %) que afectan a varias corridas seguidas; con 3
# repeticiones una mediana puede caer en ellos. Mirar siempre la tabla por corrida antes de creerle a summary.md.
set -u
SCEN=${SCEN:-decode}; REPS=${REPS:-3}; CTX=${CTX:-32768}
M=models/Inkling-2-Mini-UD-Q4_K_XL.gguf
case $SCEN in
  decode)  args=(--tokens @bench/baseline/prompts/doc-4096.txt --gen 400) ;;
  prefill) args=(--tokens @bench/baseline/prompts/doc-16384.txt --gen 8) ;;
  *) echo "SCEN: decode o prefill"; exit 2 ;;
esac
OUT=bench/ab/results/$(date +%Y%m%d-%H%M%S)-$SCEN
mkdir -p "$OUT"
export IK2_KV=${IK2_KV:-fp8} IK2_GATE_TILE=64 IK2_GATE_BF16=1 IK2_Q8_UNROLL=1 IK2_BATCH_PREFILL=1
{ echo "commit: $(git rev-parse HEAD 2>/dev/null)"; nvidia-smi --query-gpu=name,driver_version,memory.used,clocks.max.sm --format=csv,noheader
  sha256sum build/maya-ik2; echo "SCEN=$SCEN REPS=$REPS CTX=$CTX"; env | grep -E "^IK2_" | sort; printf '%s\n' "$@"; } > "$OUT/env.txt"
for r in $(seq 1 "$REPS"); do
  for cfg in "$@"; do
    name=${cfg%%=*}; vars=${cfg#*=}
    log=$OUT/$name.r$r.log
    # el estado de la GPU antes de cada corrida (comparte el escritorio): un tramo lento se ve aquí
    echo "$name r$r: $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu,memory.used --format=csv,noheader)" >> "$OUT/gpu.txt"
    env $vars build/maya-ik2 --model $M --ctx "$CTX" --chunk 2048 "${args[@]}" > "$log" 2>&1
    rc=$?
    grep -q "tokens take the resident path" "$log" || echo "!! $name r$r: no quedó residente" | tee -a "$log"
    echo "$name r$r exit=$rc $(grep -E '^(prompt|generation):' "$log" | grep -oE '\([0-9.]+ tok/s\)' | tr '\n' ' ')"
  done
done
SCEN=$SCEN python3 - "$OUT" "$@" <<'PY'
import os, re, statistics, sys, pathlib
key = "prellenado" if os.environ["SCEN"] == "prefill" else "decod."
out = pathlib.Path(sys.argv[1]); names = [c.split("=", 1)[0] for c in sys.argv[2:]]
rows = [f"| config | env | prellenado tok/s (mediana [rango]) | decodificación tok/s | Δ {key} vs 1ª |", "|---|---|---:|---:|---:|"]
base = None
for c in sys.argv[2:]:
    n, v = c.split("=", 1)
    pf, dc = [], []
    for f in sorted(out.glob(f"{n}.r*.log")):
        t = f.read_text(errors="ignore")
        m = re.search(r"^prompt: .*?\(([0-9.]+) tok/s\)", t, re.M); pf += [float(m.group(1))] if m else []
        m = re.search(r"^generation: .*?\(([0-9.]+) tok/s\)", t, re.M); dc += [float(m.group(1))] if m else []
    fmt = lambda x: f"{statistics.median(x):.1f} [{min(x):.1f}–{max(x):.1f}]" if x else "—"
    xs = pf if key == "prellenado" else dc
    md = statistics.median(xs) if xs else None
    base = md if base is None else base
    d = f"{(md / base - 1) * 100:+.1f} %" if md and base else "—"
    rows.append(f"| {n} | `{v or '(por defecto)'}` | {fmt(pf)} | {fmt(dc)} | {d} |")
(out / "summary.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
PY
echo "resultados en $OUT"
