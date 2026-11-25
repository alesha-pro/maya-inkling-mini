# A/B de variables de entorno de maya-ik2

`run_ab.sh` corre varias configuraciones alternadas (A B C A B C …), en frío y con la misma build. Por cada configuración da la mediana del prellenado y de la decodificación. Ver el encabezado del script para el uso.

**Ojo con el ruido de esta máquina.** La RTX 3090 también maneja el escritorio, y hay tramos lentos (−15 a −25 %) que afectan a varias corridas seguidas, sin importar la configuración. Con 3 repeticiones, la mediana de una configuración puede caer en un tramo lento: en la corrida de abajo, `summary.md` dio un falso +19 % por eso. Hay que mirar siempre la tabla por corrida. Desde esta versión, `gpu.txt` guarda el reloj, el consumo y la VRAM antes de cada corrida.

## Paso 3 del informe (2026-11-25, build `dd8bfe2` + `IK2_FLASH_SPLITS`, caché FP8)

Comparación usando solo las corridas en estado normal (decodificación ≥ 115 tok/s y prellenado ≥ 2400 tok/s). Datos en `results/20261125-135133-decode/` y `results/20261125-135445-prefill/`.

### SPLIT de los experts (decodificación tras 4096 tokens)

| Config | `IK2_EXPERT_MMA_SPLIT` / `_DOWN` | tok/s | Δ |
|---|---|---:|---:|
| **por defecto** | 8 / 4 | **125,0** (3 corridas) | — |
| d6 | 8 / 6 | 124,7 | −0,2 % |
| g6 | 6 / 4 | 123,9 | −0,8 % |
| d3 | 8 / 3 | 123,8 | −0,9 % |
| g6d6 | 6 / 6 | 123,5 | −1,2 % |
| g6d3 | 6 / 3 | 122,4 | −2,0 % |
| g4 | 4 / 4 | 120,5 | −3,5 % |

**El valor por defecto (8/4) es el más rápido.** Que los bloques de K se repartan exacto (6/6) no compensa tener menos warps en vuelo. El orden se repite igual en las dos repeticiones buenas. Esto contradice la estimación del informe (+1–3 %).

### Splits de la atención flash (prellenado de 16 384 tokens)

| Config | `IK2_FLASH_SPLITS` | tok/s | Δ |
|---|---|---:|---:|
| **heurística** (`attn_flash_splits`) | — | **2715** (1 corrida) | — |
| ns2 | 2 | 2726 | +0,4 % |
| ns4 | 4 | 2712 | −0,1 % |
| ns8 | 8 | 2679 | −1,3 % |
| ns16 | 16 | 2608 | −4,0 % |
| ns1 | 1 | 2594 | −4,5 % |

**La heurística ya está en el óptimo**: ninguna cantidad fija la supera por encima del ruido (±3 %). El "+10–30 % en las visitas globales" del informe no aparece. La base tiene una sola corrida buena; si hiciera falta confirmarlo, habría que repetir solo base frente a ns2.

### `cand_topk` en un lanzamiento

No se midió. Son 3 kernels pequeños por fila, y solo cuando se muestrea dentro de una ventana del drafter. Lo estimo en ~0,2 % de una ventana de ~20 ms, muy por debajo del ruido con drafter (±10 %), así que no se podría demostrar una ganancia.

### Paridad

Con `IK2_FLASH_SPLITS=2` (el "ganador" automático, que resultó ser ruido) los tres modos del control pasan: top-1 98,1 / 97,7 / 98,2 %. Cambiar el número de splits no rompe la exactitud.
