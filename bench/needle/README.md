# Needle-in-a-haystack de Inkling-2-Mini (paso 4 del informe)

`tools/needle_bench.py` contra el servidor en marcha: esconde una palabra clave en un texto largo (los docs y el código del repo) y la pide por la API, con greedy y sin razonamiento.

```bash
python3 tools/needle_bench.py --url http://127.0.0.1:18000 --lengths 32k,64k,128k --depths 10,50,90 --out bench/needle/results/X.json
```

Cambios a la herramienta en esta versión:
- **Saca `needle_bench.py` del pajar.** Ese archivo contiene la frase de la aguja sin palabra clave, así que un pajar armado con `tools/*.py` llevaba un señuelo.
- **Usa 3,1 caracteres por token en vez de 3,2.** Medido con el tokenizador de Inkling sobre este pajar: 3,13–3,17. Con 3,2, "128k" daba 131 074 tokens, que no entran en un contexto de 131 072.

## 2026-11-25: 9 de 9

Servidor `docs/inkling2-mini/ik2-mini-128k.json`: contexto 131 072, caché KV FP8, drafter, residente. Build de la rama `ik2-parity-gate` (`dd8bfe2`). Resultados en `results/20261125-112724-server-fp8-128k.json`.

| Longitud (tokens) | 10 % | 50 % | 90 % | Tiempo por pedido |
|---|---|---|---|---:|
| 31 668 | ✓ | ✓ | ✓ | 13–14 s |
| 62 531 | ✓ | ✓ | ✓ | 27–28 s |
| 127 377 | ✓ | ✓ | ✓ | 67–75 s |

En los 9 casos la respuesta fue exactamente la palabra clave, sin texto de más. El tiempo incluye el prellenado completo: el pedido de ~127K corre a ~1700–1900 tok/s.

## Qué no dice esto

- **Recuperar una palabra textual es lo más fácil del contexto largo.** No mide razonamiento sobre todo el texto ni varios saltos; para eso haría falta una tarea tipo RULER.
- **No se midió más allá de 128K.** 344 064 (la doc) no entra en VRAM en esta máquina, porque la GPU comparte el escritorio; el máximo residente es 315 392. El perfil 1M sigue sin verificar.
- **No se midió la deriva de las probabilidades con la posición.** La referencia float64 en CPU no es viable a esas longitudes. Queda pendiente medir FP8 frente a FP16 por tramos de contexto (4K/32K/64K/120K), con el servidor parado.
