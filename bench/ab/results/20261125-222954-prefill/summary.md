| config | env | prellenado tok/s (mediana [rango]) | decodificación tok/s | Δ prellenado vs 1ª |
|---|---|---:|---:|---:|
| c4k | `CHUNK=4096` | 3055.3 [2873.1–3063.2] | 99.2 [99.2–106.0] | +0.0 % |
| pf64 | `IK2_PF_NVFP4=1 IK2_PF_BM=64 CHUNK=4096` | 3022.5 [2981.5–3089.7] | 104.9 [100.4–107.3] | -1.1 % |
| pf128 | `IK2_PF_NVFP4=1 CHUNK=4096` | 2947.6 [2892.6–3074.4] | 105.9 [96.8–107.9] | -3.5 % |
