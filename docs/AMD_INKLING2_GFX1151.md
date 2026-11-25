# Inkling 2 Mini on AMD gfx1151 (Strix Halo)

What the AMD build of `maya-ik2` adds for Ryzen AI Max (gfx1151, Radeon 8060S, one RAM for CPU and GPU), and how it
was checked. Everything is under `#if defined(STRATA_USE_HIP)`: the CUDA build reads the same sources it read before
(`python tools/nv_path_check.py` proves it as text).

## Build

    cmake -B build-hip -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_HIP_IK2_SPIKE=ON \
          -DCMAKE_HIP_ARCHITECTURES=gfx1151
    cmake --build build-hip --target maya-ik2 ik2_kernels_test ik2_fuse_test ik2_experts_test

The tensor-core (PTX) paths are not in the AMD build: their settings are refused, and so is `--parallel > 1`
(its batched attention is one of them).

## What is on by default, and where

The tuned paths are on by default only on **gfx1151 with one GPU** (where they were measured). Every other AMD
device keeps the plain path unless the settings ask; the prefill kernels (`IK2_HBPF`) need gfx11 (WMMA).

| class | setting (gfx1151 default) | exact? | the check |
| --- | --- | --- | --- |
| load | (always) pinned bounce ring for the experts' copy-in | yes | the same tokens |
| fused decode | `IK2_FUSE_MASK=0xf14f`, `IK2_GRAPH=0` | yes, bit for bit | `ik2_fuse_test`, `ik2_experts_test` (memcmp); greedy ids on vs off |
| prefill | `IK2_HBPF=183` | no (another sum order) | KL divergence vs an fp64 reference, as near as the old path |
| memory | `IK2_MEMTRIM=0x13` | yes | greedy ids on vs off; GTT peak |
| DSpark window | `IK2_DSC` (off) | yes | `IK2_DSC_PARITY`, `ik2_dsc_test` |

## Measured

`--bench`'s measurement (3 questions x 256 tokens, greedy, thinking off, one warm-up; prefill at 2048 / 8192), driven
over `--serve`, context 16,384, `Inkling-2-Mini-UD-Q4_K_XL.gguf`, gfx1151 (Radeon 8060S, 32 GB shared), ROCm 7.2.4.
`main` does not build on AMD, so "before" is this branch with every tuned path off (`IK2_FUSE_MASK=0 IK2_GRAPH=1
IK2_HBPF=0 IK2_MEMTRIM=0`). Two runs each (night, idle / day, desktop in use):

| | before (tuned paths off) | after (defaults) |
| --- | --- | --- |
| decode, mean of 3 | 38.7 / 39.9 tok/s | 46.2 / 46.2 tok/s |
| prefill 2048 | 243 / 314 tok/s | 1,356 / 1,616 tok/s |
| prefill 8192 | 232 / 299 tok/s | 1,357 / 1,506 tok/s |

Prefill answers: KL divergence against an fp64 reference (4096-token prompt, last 512 positions) 2.764e-3 with
`IK2_HBPF=183` vs 2.913e-3 on the old path; paired difference -1.5e-4, SE 2.0e-4.

## Credits and third-party code

- `src/ik2/ik2_hbpf.inc` ports prefill kernels from **headbouyJB/inkling2-gufo** (commit 8bb6cd5, `src/hip/fast.hip`):
  `k_gemm4` and its dequant helpers, `k_sum_pairs`, `k_hist` / `k_scan_items` / `k_scatter`, `attn_prefill`, and
  the host wrappers' launch shapes. The changes are listed at the top of that file.
- `k_rg` (`IK2_HBPF` bit 7) follows the shape of **gufo**'s routed expert GEMMs (gufo-org/gufo v0.35
  `RoutedDownF16Kernel` / `RoutedPairF16Kernel`, `SwapHalves`), with the NVFP4 type and 192-token tile of
  mikealanni/gufo's `RoutedF16GEMMKernel<kNVFP4>`.
- The engine itself is Maya's (and Strata's `hip_compat` layer).
- Made by Claude (Anthropic) working for Cold-Serve Labs.

headbouyJB/inkling2-gufo's license:

```
MIT License

Copyright (c) 2026 headbouyJB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


gufo's license:

```
MIT License

Copyright (c) 2026 gufo contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

