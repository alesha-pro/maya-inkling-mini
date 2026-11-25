| config | env | prellenado tok/s (mediana [rango]) | decodificación tok/s | Δ prellenado vs 1ª |
|---|---|---:|---:|---:|
| base | `(por defecto)` | 2578.4 [2440.3–2631.1] | 101.2 [97.0–102.9] | +0.0 % |
| c4k | `CHUNK=4096` | 3090.1 [3064.2–3187.7] | 112.2 [111.8–112.2] | +19.8 % |
| nodec | `IK2_EXP_SKIP_DECODE=1` | 3831.5 [3737.8–3886.3] | 113.0 [105.0–113.1] | +48.6 % |
| nodec4k | `IK2_EXP_SKIP_DECODE=1 CHUNK=4096` | 3932.9 [3847.3–3980.7] | 107.6 [95.7–108.7] | +52.5 % |
