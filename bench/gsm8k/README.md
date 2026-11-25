# GSM8K de Inkling-2-Mini (paso 5 del informe)

Una sola receta, contra el servidor en marcha:

```bash
python3 tools/gsm8k_bench.py run --url http://127.0.0.1:18000 --label fp8-drafter \
    --out bench/gsm8k/results/$(date +%Y%m%d-%H%M%S)-fp8-drafter.json
python3 tools/gsm8k_bench.py compare bench/gsm8k/results/A.json bench/gsm8k/results/B.json
```

- **Preguntas:** el `test.jsonl` de `openai/grade-school-math` en el commit `b0bb162` (las 1319, licencia MIT). Se baja una vez a `data/gsm8k/`, que no se versiona, y se comprueba su SHA-256 (`3730d312…`). `--n 96` corre solo las 96 primeras, para una mirada rápida.
- **Pedido:** cada pregunta va sola como turno de usuario a `/v1/chat/completions`, con razonamiento apagado, `temperature 0` y `max_tokens 1024`. El prompt pide los pasos y la respuesta final en la última línea como `#### <número>`, y es fijo (`PROMPT` en el script).
- **Corrección:** gana el número tras el último `####`, o si no hay marca, el último número de la respuesta. Se normalizan `$`, comas y `.0`. Una respuesta cortada por `max_tokens` cuenta como mal y queda marcada `length`. Con las 1319 respuestas oficiales, el extractor da el número correcto en todas.
- **Salida:** el JSON guarda el bloque `engine` de `/metrics` (KV, contexto, motor), la receta, cada respuesta y su veredicto. Así dos corridas se comparan pregunta por pregunta: `compare` hace McNemar exacto (binomial sobre los pares discordantes).

## Por qué la doc dice 74 en un lado y 89–90 en otro

Las dos cifras salen de 96 preguntas, pero de arneses distintos que no están en el repo:

| Cifra | Dónde | Quién | Lo que se sabe de la receta |
|---|---|---|---|
| 73 / 74 / 77 de 96 | `docs/INKLING2-MINI.md` (calidad), README | autor del repo, 2026-11-21 | greedy, razonamiento apagado. No figuran el prompt, la caché, qué 96 preguntas ni `max_tokens`. |
| 89 / 90 de 96 | PR #5 (`--parallel`), `docs/INKLING2-MINI.md` | autor del PR #5, 2026-11-21 | greedy, `IK2_KV=fp8`, sin drafter, `--max-context 32768`, por el servidor; 30 725 tokens generados, unos 320 por respuesta. No figuran el prompt, qué 96 ni el extractor. |

La diferencia (15 de 96) es demasiado grande para ser la caché o el drafter: FP8 frente a FP16 empeora el top-1 en ~2 puntos en el control de paridad, y el drafter no cambia la distribución. Lo que más probablemente la explica es el arnés: otras preguntas, otro prompt, otro límite de tokens u otro extractor (un extractor de "último número" sin marca, o respuestas cortadas, pueden mover fácilmente 10–15 puntos). **No se puede saber cuál**: no hay respuestas por pregunta guardadas de ninguna de las dos. Por eso las dos cifras no son comparables entre sí, y tampoco con las de esta receta. Desde ahora, toda cifra de GSM8K debería salir de este script, con su JSON en `results/`.

## Resultados

### 2026-11-25: 1168 de 1319 (88,5 %)

Servidor `docs/inkling2-mini/ik2-mini-128k.json`: caché FP8, drafter, contexto 131 072, `--parallel 1`. Build `dd8bfe2`, del motor de antes del refactor del muestreo. Datos en `results/20261125-142712-server-fp8-drafter.json`.

| | |
|---|---:|
| Correctas | **1168 / 1319 = 88,5 %** (Wilson 95 %: 86,7–90,2) |
| Cortadas por `max_tokens` 1024 | 13 (cuentan como mal; algunas se quedan dudando o se repiten) |
| Sin número | 0 |
| Sin la marca `####` | 2 |
| Tokens generados | 153 112 (mediana 88 por respuesta) |
| Tiempo | 21 min |

**Las 96 primeras: 88 de 96.** Con esta receta, la cifra queda junto al 89/90 del PR #5 (también FP8) y lejos del 74. Eso apoya que el 74 salió de otro arnés y no de la caché ni del motor, aunque no prueba cuál era.

### 2026-11-25: FP16 frente a FP8, 1185 frente a 1168 (p = 0,10)

El mismo servidor con `IK2_KV=f16` y `--max-context 106496`, que es lo que entra en VRAM con el drafter. Lo demás es igual. Datos en `results/20261125-170500-server-f16-drafter.json`.

| Caché | Correctas | Wilson 95 % | Cortadas | Las 96 primeras |
|---|---:|---|---:|---:|
| FP8 (128K) | 1168 (88,5 %) | 86,7–90,2 | 13 | 88 |
| FP16 (104K) | 1185 (89,8 %) | 88,1–91,4 | 11 | 87 |

`compare`: 39 preguntas solo bien con FP8 y 56 solo bien con FP16. **McNemar exacto: p = 0,10.** FP16 va 1,3 puntos arriba, en la misma dirección que el control de paridad (top-1 −1,8 puntos, KL ×3 con FP8), pero no es significativo con 1319 preguntas. Lo que sí muestra el par: 95 preguntas (7 %) cambian de resultado solo por la caché. Con greedy, una diferencia de una o dos preguntas entre dos corridas no dice nada. En las 96 primeras, FP16 da 87 y FP8 88: el orden se invierte, otra señal de lo poco que distinguen 96 preguntas.
