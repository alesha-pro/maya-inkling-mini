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

## Paso 6 del informe: de dónde sale el tiempo del prellenado (2026-11-25)

Prellenado de 16 384 tokens, caché FP8, build `build-s5` (la rama del paso 5 más `IK2_EXP_SKIP_DECODE`, que es solo de medición), 3 repeticiones alternadas. Datos en `results/20261125-213535-prefill/`.

| Config | tok/s (mediana [rango]) | Δ |
|---|---:|---:|
| base (`--chunk 2048`) | 2578 [2440–2631] | — |
| `--chunk 4096` | 3090 [3064–3188] | **+19,8 %** |
| sin decodificar los experts a FP16 (salida incorrecta, solo cota) | 3832 [3738–3886] | +48,6 % |
| lo mismo con `--chunk 4096` | 3933 [3847–3981] | +52,5 % |

- **Un tercio del prellenado es decodificar los experts NVFP4 a FP16** antes de cada GEMM de cuBLAS (`run_prefill_moe`: `decode_f16` del expert entero, en cada una de las 48 visitas y cada chunk). Un GEMM que lea NVFP4 directamente, decodificando en registros o en memoria compartida, puede ganar como máximo +49 % con chunk 2048 y +27 % sobre chunk 4096. Con el costo de decodificar dentro del kernel, la ganancia real va a ser menor.
- **`--chunk 4096` da +20 % sin tocar código**, porque la decodificación se hace una vez por chunk y se reparte entre el doble de tokens. Con la config del servidor (131 072, FP8, drafter, `IK2_DRAFT_CTX=65536`), los experts siguen residentes: 16K se prellenó a 2942 tok/s en esa prueba suelta. Con 3072, 2668 tok/s. Cambia el redondeo del prompt (otro chunk puede dar otra respuesta greedy), como cualquier cambio de chunk.

### `--chunk 4096` en el servidor y un GEMM NVFP4 directo (`IK2_PF_NVFP4`), 2026-11-25

`--chunk 4096` quedó en `ik2-mini-128k.json`, la config local del servidor. Sigue residente, y el needle tarda 12 s a 32K (antes 13–14) y 24 s a 64K (antes 27–28).

`k::experts_pf_nvfp4` es un GEMM agrupado: todos los experts de una visita en un lanzamiento por sentido. Los pesos NVFP4 se decodifican en memoria compartida con la aritmética exacta de `decode_f16`, y el producto va por `mma.sync` con sumas F32. `tests/ik2_kernels_test.cu` lo compara contra una referencia en double: 5 % de la tolerancia, con tiles de 64 y de 128 filas. El control de paridad, en modo `dump` con `IK2_BATCH_PREFILL=1` (el camino del servidor; el control no lo usaba), pasa igual o mejor: top-1 98,5 % frente a 97,8 %, KL media 2,7e-3 frente a 3,1e-3.

| Prellenado 16K | tok/s | Δ |
|---|---:|---:|
| chunk 2048, decode + cuBLAS | 2504 | — |
| chunk 2048, `IK2_PF_NVFP4=1` | 2982 | **+19 %** |
| chunk 4096, decode + cuBLAS | 2929 / 3055 (2 sesiones) | — |
| chunk 4096, `IK2_PF_NVFP4=1` | 2944 / 2948 | +0,5 % / −3,5 % (ruido) |
| chunk 4096, tiles de 64 filas | 3023 | −1,1 % (ruido) |

Con chunk 2048 el kernel gana lo mismo que chunk 4096. Con chunk 4096 empatan: el kernel se ahorra la decodificación, pero multiplica más lento que cuBLAS. Las tiles de 128 filas, que decodifican la mitad, no cambian nada, así que el cuello es su aprovechamiento de los tensor cores. Queda apagado por defecto. Sirve donde chunk 4096 no entra en VRAM.
