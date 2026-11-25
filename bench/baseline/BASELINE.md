# Línea base de maya-ik2 — paso 0 (2026-11-25)

**Build:** HEAD `cc41ab1`, `build/maya-ik2` sha256 `8bbdb8d8…b675b3`, CUDA 12.8, sm_86, Release (ninja: sin cambios que recompilar).
**Máquina:** RTX 3090 24 GB (driver 615.78.08, límite 370 W; **la misma GPU maneja el escritorio, ~0,5 GiB ocupados**), 12 hilos, 31 GiB RAM. Contenedor `maya-ik2-dev:latest`.
**Entorno:** `IK2_KV=fp8 IK2_GATE_TILE=64 IK2_GATE_BF16=1 IK2_Q8_UNROLL=1 IK2_BATCH_PREFILL=1` (drafter: `IK2_DRAFT_CTX=65536`).
**Prompts fijos:** `bench/baseline/prompts/doc-{512,4096,16384}.txt`, que son los primeros N tokens de `docs/INKLING2.md` + `docs/SETTINGS.md`. Decodificación greedy de 400 tokens (384 con drafter). Cada fila es un arranque en frío, con 3 repeticiones.

**Reproducir** (desde la raíz del repo, con la GPU libre):

```bash
docker run --rm --gpus all -u 1000:1000 -e HOME=/tmp -e TAG=residente -e CTX_PLAIN=315392 -e CTX_DRAFT=241664 \
  -v "$PWD":/work -w /work --entrypoint bash maya-ik2-dev:latest -lc 'bench/baseline/run_baseline.sh 3'
```

## Resultados (camino residente, `results/20261125-040934-residente/`)

Mediana de 3 repeticiones; el rango va entre corchetes.

| Escenario | Contexto | Prellenado, tok/s | Decodificación, tok/s |
|---|---:|---:|---:|
| 512 tokens, sin drafter | 315 392 | 1048 [1017–1049] | 122,7 [122,6–123,1] |
| 4096 tokens, sin drafter | 315 392 | 2489 [2467–2593] | 119,4 [119,3–120,0] |
| 16 384 tokens, sin drafter | 315 392 | 2545 [2536–2577] | 113,2 [110,8–113,7] |
| 512 tokens, con drafter | 241 664 | 985 [968–1007] | 126,5 [110,5–130,2] |
| 4096 tokens, con drafter | 241 664 | 2509 [2486–2551] | 154,3 [153,4–161,3] |

Drafter (repetición 1): con 512 tokens, 1,92 tokens por ventana; con 4096 tokens, 3,48 tokens por ventana. En ambos casos una ventana cuesta ~20 ms y un token normal ~8,8 ms.

### Comparación con la documentación (`docs/INKLING2-MINI.md`, 3090 dedicada a 300 W, contexto 344 064)

| | Doc | Medido | Δ |
|---|---:|---:|---:|
| Prellenado 4096 | 2742 | 2489 | −9 % |
| Decodificación tras 4096 | 124,2 | 119,4 | −4 % |

Hay diferencias de máquina sin aislar: la GPU comparte el escritorio, el driver es otro, la CPU no es EPYC y el prompt es otro texto. **Usar estos números, no los de la doc, como referencia para los A/B.** El ruido entre repeticiones es ~±1 % en la decodificación sin drafter y ~±3 % en el prellenado; con drafter llega a ±10 %, porque depende del texto.

## Hallazgo: las configs de referencia no son residentes en esta máquina

La primera pasada (`results/20261125-040602-no-residente/`) usó los contextos de la doc y el motor **no quedó residente**:

- Contexto 344 064 sin drafter: "the expert cache holds 3042 of 3072 experts (0.14 GiB short)". El motor sugiere `--max-context 315392`.
- Contexto 262 144 con drafter (= `ik2-mini.json`): "0.04 GiB short … the drafter is off". El motor sugiere `--max-context 241664`.

Resultado: 77–81 tok/s de decodificación (≈ −35 %), prellenado de 4096 tokens a ~1050 tok/s (−58 %), y **con drafter fue 44–63 tok/s, peor que sin él**. Con el escritorio en la misma 3090, `ik2-mini.json` tal como está (262 144) cae en este camino lento. El servidor actual usa `ik2-mini-128k.json` (131 072) y no se ve afectado.

## Pendiente

- No hay perfil por kernel: `nsys` no está en la imagen y `ncu` sí, pero no se corrió. Es el siguiente paso para validar las estimaciones de §1.
- Tampoco se midió nada a través del servidor HTTP, ni con `--parallel`, ni con prompts de 64K o más.

## Intento de perfil por kernel (2026-11-25): cancelado

- **`IK2_PROF=1 IK2_STATS=1`** (4096 tokens + 200): `profile (all calls): attention 1.41 s, router 0.32 s, uploads 0.01 s, experts 0.69 s, rest 0.22 s`. El perfilado sincroniza en cada fase y la decodificación cae a 74 tok/s, así que solo sirve para ver proporciones, no tiempos absolutos. Atención + router suman ~65 % del tiempo medido, más que los experts. Log en `profile/ik2_prof.log`.
- **`ncu --graph-profiling node`** se colgó: en 30 min no perfiló ningún kernel, ocupó toda la VRAM (23,9 GiB) y, como la GPU maneja el escritorio, la PC se trabó. Se canceló.
  **No repetir en esta GPU con el escritorio activo.** Alternativas: otra GPU o sesión sin escritorio, `nsys` (no está en la imagen), o cronometrar kernels con eventos CUDA dentro del motor.
