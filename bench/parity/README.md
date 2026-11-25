# Control de paridad de maya-ik2

Compara las log-probabilidades del motor con las de la referencia float64 (`ref/inkling2.py`) sobre prompts fijos. Si se rompe un umbral, falla (código de salida 1). Sirve como nightly en GPU o como paso previo a un merge.

| Archivo | Qué es |
|---|---|
| `prompts/*.txt` | 4 prompts de 512 tokens (prosa, Python, CUDA, chino): 2048 posiciones en total |
| `ref/*.npz` | Referencia compacta: el top-256 de log-probabilidades por posición (se versiona) |
| `MANIFEST.sha256` | Hashes de prompts y referencias; `run_gate.sh` se niega a correr si no coinciden |
| `make_ref.sh` | Genera las referencias en CPU (~15–30 min por prompt). Solo hace falta si cambia el modelo o la referencia |
| `run_gate.sh` | El motor en GPU (caché KV en FP16) en tres modos + `tools/ik2_compare.py --gate` por modo → `results/*-<modo>.json` |

```bash
PY=~/.cache/maya-ref-venv/bin/python bench/parity/make_ref.sh   # una vez (venv con torch CPU + gguf-py)
bench/parity/run_gate.sh                                        # cada cambio del motor; GPU libre (~3 min)
MODES=decode bench/parity/run_gate.sh                           # un solo modo
```

## Modos

| Modo | Qué camino del motor mide | Posiciones |
|---|---|---|
| `dump` | `--dump`: el camino general (prellenado por trozos), y capa por capa si está la referencia completa | 4 × 512 |
| `decode` | Prellenado de `K=128` tokens (supercapas) y luego los otros 384 **de uno en uno por el camino residente** (`fast_token` + grafo CUDA), forzados con `IK2_FORCE`; logits de `IK2_LOGITS_OUT`. Es lo que corre el servidor al decodificar | 4 × 384 |
| `block` | Igual, pero en **ventanas de `W=5` por el camino de bloques** (`fast_block`, `IK2_FORCE_BLOCK`): la ventana de verificación del drafter | 4 × 384 |

`decode` y `block` comprueban en el log que el motor quedó residente; si no, salen con 2. `ik2_compare.py` lee los logits crudos (`logits.f32`) y compara desde la posición de `first_pos`.

## Primera corrida (2026-11-25, HEAD `cc41ab1`, caché KV FP16): PASS

| Prompt | top-1 | KL media |
|---|---:|---:|
| cuda | 96,7 % | 5,3e-3 |
| prosa | 97,3 % | 1,9e-3 |
| python | 97,5 % | 4,4e-3 |
| zh | 99,6 % | 8,3e-4 |
| **Total (2048 posiciones)** | **97,8 % (IC95% 97,0–98,3)** | **3,1e-3** (p99 3,0e-2, máx. 0,18) |

Hubo 46 cambios de token. 39 son casi-empates (algún lado con margen < 0,1 nats) y 7 son "reales", con márgenes ≤ 0,36 nats. Revisé los 7 reales y el de mayor margen de la referencia: en todos, ese mismo token ya había elegido expertos distintos en alguna visita anterior. Son **cascadas de casi-empates del router** a través del loop, no un kernel roto. Ejemplo: cuda, posición 377. El token es idéntico hasta la visita 8 (cos 1,0000); en la visita 9 cambia un experto y el motor termina dudando entre los dos tokens (0,212 frente a 0,207).

Resumen completo en `results/20261125-053707-cc41ab1.json`.

## Segunda corrida (2026-11-25, `cc41ab1` + `IK2_FORCE_BLOCK`, caché KV FP16): los tres modos PASS

| Modo | Posiciones | top-1 (IC95%) | Cambios (casi-empates / reales) | KL media / p99 / máx. |
|---|---:|---|---|---|
| dump | 2048 | 97,8 % (97,0–98,3) | 46 (39 / 7) | 3,1e-3 / 3,0e-2 / 0,18 |
| decode | 1536 | 98,2 % (97,4–98,7) | 28 (26 / 2) | 3,2e-3 / 3,5e-2 / 0,11 |
| block | 1536 | 98,4 % (97,7–98,9) | 24 (22 / 2) | 3,0e-3 / 3,2e-2 / 0,11 |

`dump` repitió la primera corrida al dígito: con caché FP16 es determinista. El camino residente y el de bloques tienen **la misma paridad que el general**, así que los umbrales sirven para los tres. Los 2 + 2 cambios "reales" de decode y block no los revisé uno por uno; tienen el mismo margen bajo que los de `dump`. Resúmenes en `results/20261125-121143-cc41ab1-{dump,decode,block}.json`.

### El control detecta una degradación real: caché FP8 (la del servidor)

El mismo modo `decode` con `IK2_KV=fp8`, que es lo que usa `docs/inkling2-mini/ik2-mini-128k.json`, **falla**:

| | FP16 | FP8 | |
|---|---:|---:|---|
| top-1 | 98,2 % | 96,0 % (94,9–96,9) | |
| cambios reales | 2 | 20 | > 15 ✗ |
| KL media | 3,2e-3 | 1,0e-2 | ×3,2 ✗ |
| KL p99 / máx. | 3,5e-2 / 0,11 | 8,7e-2 / 0,48 | ✗ / ✗ |
| máx. \|Δlogp\| | 0,51 | 1,49 | ✗ |

Es una sola corrida (FP8 no es determinista entre corridas) y son solo 512 tokens de contexto. Aun así es la primera medida del costo de calidad de la caché FP8 con un tamaño de muestra suficiente. Resumen en `results/20261125-121412-cc41ab1-decode-fp8.json`. Para vigilar FP8 hacen falta umbrales propios (pasar las flags de límite a mano); los de por defecto son de FP16.

## Umbrales (`tools/ik2_compare.py`, por defecto) = control de regresión

Son lo medido arriba más un margen. El control falla si el motor empeora respecto a hoy.

| Métrica | Límite | Medido |
|---|---|---|
| posiciones | ≥ 1000 | 2048 |
| top-1 | ≥ 0,96 | 0,978 |
| cambios reales (ambos lados ≥ 0,1 nats) | ≤ 15 | 7 |
| KL media / p99 / máx. | 6e-3 / 6e-2 / 0,4 | 3,1e-3 / 3,0e-2 / 0,18 |
| máx. \|Δlogp\| del top de la referencia | 1,0 | 0,56 |

Cada umbral se puede cambiar con su flag. Si un cambio del motor mejora la paridad de forma estable, hay que bajar los límites.

## Límites conocidos

- `decode`/`block` miden la salida `kLogits`. El servidor pide `kArgmax`/`kCand`, que son otros grafos CUDA con los mismos kernels salvo el final. Tampoco cubren el **retroceso tras rechazar borradores** (`block_rollback`), `--parallel` (`fast_slots`) ni contextos largos (todo pasa en < 512 posiciones). Las supercapas solo se cubren de forma indirecta, por la caché KV de los 128 tokens de prellenado.
- **Seam `attn_out` desalineado en las capas sliding:** sale con error relativo ~1–5 en todas las capas menos las full (0, 6, 12…), aunque la salida de esas capas coincide (error ~1e-3..1e-1). Es casi seguro que motor y referencia vuelcan puntos distintos con ese nombre. Hay que alinearlos antes de usar ese seam para diagnosticar.
- Con una referencia compacta, el KL se calcula sobre el top-256 de la referencia. La masa cubierta se reporta.
