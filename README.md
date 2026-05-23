# PRSD — Parallel Ring Streaming Dataflow for CNN Convolution Acceleration on FPGA

> HLS project comparing **Ring Streaming Dataflow (RSD)** and the proposed  
> **Parallel Ring Streaming Dataflow (PRSD)** for 2-D convolution acceleration.  
> PRSD achieves approximately **1.9× lower latency** by computing two output rows  
> simultaneously, with a corresponding increase in Hardware utilisation.

---

## Table of Contents

1. [Background and motivation](#1-background-and-motivation)
2. [Dataflow architectures](#2-dataflow-architectures)
   - 2.1 [Normal streaming dataflow](#21-normal-streaming-dataflow)
   - 2.2 [Ring streaming dataflow (RSD)](#22-ring-streaming-dataflow-rsd)
   - 2.3 [Parallel ring streaming dataflow (PRSD)](#23-parallel-ring-streaming-dataflow-prsd)
   - 2.4 [Side-by-side comparison](#24-side-by-side-comparison)
3. [Loop tiling](#3-loop-tiling)
4. [Code architecture](#4-code-architecture)
   - 4.1 [RSD function hierarchy](#41-rsd-function-hierarchy)
   - 4.2 [PRSD function hierarchy](#42-prsd-function-hierarchy)
   - 4.3 [Top-level calling pattern](#43-top-level-calling-pattern)
5. [HLS pragma strategy](#5-hls-pragma-strategy)
6. [CNN model architectures tested](#6-cnn-model-architectures-tested)
   - 6.1 [LeNet-5](#61-lenet-5)
   - 6.2 [Mini VGGNet-32](#62-mini-vggnet-32)
7. [Performance model](#7-performance-model)
   - 7.1 [Tile operation count formula](#71-tile-operation-count-formula)
   - 7.2 [GOPS formula](#72-gops-formula)
   - 7.3 [PRSD vs RSD speedup derivation](#73-prsd-vs-rsd-speedup-derivation)
   - 7.4 [Worked examples — all layers](#74-worked-examples--all-layers)
8. [Verified results table](#8-verified-results-table)
9. [Hardware utilisation tradeoff](#9-hardware-utilisation-tradeoff)
10. [File structure](#10-file-structure)
11. [How to run](#11-how-to-run)
12. [References](#12-references)

---

## 1. Background and motivation

Convolutional Neural Networks (CNNs) require billions of multiply-accumulate (MAC)
operations per inference pass. On FPGAs, efficiency is tightly coupled to how data
is streamed through the processing array: poor streaming creates pipeline bubbles and
leaves compute resources idle at row boundaries.

This project extends the work presented in:

> *ESSA: An Energy-Aware Bit-Serial Streaming Deep Convolutional Neural Network
> Accelerator*

That paper introduced the **Ring Streaming Dataflow (RSD)** and combined it with loop
tiling, kernel decomposition, and bit-serial arithmetic to build an efficient CNN
accelerator. This project focuses on improving the dataflow layer only.

The proposed **Parallel Ring Streaming Dataflow (PRSD)** extends RSD by capturing one
additional IFM row per outer-loop iteration, enabling **two OFM rows to be produced
simultaneously per scan pass**. This approximately halves the number of H-loop
iterations and achieves **~1.9× lower latency** across all tested layers, at the cost
of increased DSP and LUT utilisation.

The improvement is validated on convolution layers from two CNN models:

- **LeNet-5** — lightweight model, 5×5 kernels, single/low channel counts
- **Mini VGGNet-32** — deeper model, 3×3 kernels, up to 64 channels

---

## 2. Dataflow architectures

### 2.1 Normal streaming dataflow

In a conventional sliding-window convolution the kernel scans each row left to right,
then **resets to column 0** before starting the next row. Every pixel in the trailing
kernel columns is discarded at the row boundary and must be re-fetched, wasting memory
bandwidth and adding latency.

### 2.2 Ring streaming dataflow (RSD)

RSD eliminates the column-reset penalty by making the sliding window **snake** across
rows in a boustrophedon (alternating-direction) pattern:

- Even-indexed rows: scan **left → right**
- Odd-indexed rows: scan **right → left**
- At every row boundary: **shift down by one row — no column reset**

Each IFM pixel stays in the kernel window for the maximum possible number of output
positions before eviction, minimising redundant data reads.

**Output rate:** one OFM pixel per innermost-loop iteration.

**HLS loop structure:**

```
H : K-1 → CONV_IN_HEIGHT         (outer row loop, step = CONV_STRIDE)
  if H even:
    W : K-1 → CONV_IN_WIDTH      (left → right)
  else:
    W : CONV_IN_WIDTH-K → 0      (right → left)
      C : 0 → CONV_OUT_CHANNELS
        ks : 0 → CONV_IN_CHANNELS
          i, j : 0 → K
            local += IFM[ks][...][...] * F[C][ks][i][j]
          OFM[C][H+1-K][W+1-K] += local
```

### 2.3 Parallel ring streaming dataflow (PRSD)

PRSD builds on RSD by capturing **one extra IFM row** per outer H step, so that one
scan pass accumulates partial results for **two output rows simultaneously**.

The outer H loop advances by **4** (instead of 1). Four local accumulators
`local_ofm_0 / 1 / 2 / 3` replace the single scalar. After completing the inner
channel and kernel loops, two OFM rows are written unconditionally; two additional
rows are written conditionally at spatial boundaries where the extended window
overlaps valid output positions.

PRSD uses two outer H loops:

| Loop | H start | H step | W direction | OFM rows produced |
|------|---------|--------|-------------|-------------------|
| Forward (`ring_conv`) | `K` | +4 | left → right | ofm_0, ofm_1 always; ofm_2, ofm_3 at right boundary |
| Reverse | `K+2` | +4 | right → left | ofm_0, ofm_1 always; ofm_2, ofm_3 at left boundary |

**Edge-case handling:** At width boundaries and height boundaries, partial results for
the 3rd and 4th rows of the extended window are conditionally accumulated via boolean
guards (`is_width_boundary`, `filter_pos_valid_k2/k3`, height bound checks), each
mapped to LUT resources in hardware.

**HLS loop structure (forward pass):**

```
H : K → CONV_IN_HEIGHT+2p, step +4
  #pragma HLS PIPELINE II=1 rewind
  W : K-1 → CONV_IN_WIDTH+2p
    #pragma HLS UNROLL factor=15
    C : 0 → CONV_OUT_CHANNELS_COM
      local_ofm_0 = local_ofm_1 = local_ofm_2 = local_ofm_3 = 0
      ks : 0 → CONV_IN_CHANNELS
        #pragma HLS PIPELINE II=1
        case-1 (W==K-1, H!=K):
          partial left-edge accumulation → local_ofm_0, local_ofm_1
        case-2 (all other W):
          i, j : 0 → K
            #pragma HLS UNROLL factor=3
            if width_bounds_valid:
              mult_result_n = IFM[ks][...] * F[C][ks][i][j]
              local_ofm_n += mult_result_n  (n=0,1,2,3 with bounds guards)
      OFM[C][H-K][W-K+1]   += local_ofm_0   (unconditional)
      OFM[C][H-K+1][W-K+1] += local_ofm_1   (unconditional)
      if is_width_boundary && is_not_height_boundary:
        OFM[C][H-K+2][W-K+1] += local_ofm_2
        OFM[C][H-K+3][W-K+1] += local_ofm_3
```

### 2.4 Side-by-side comparison

| Aspect | RSD | PRSD |
|--------|-----|------|
| H loop step | +1 (stride) | +4 |
| OFM rows written per H pass | 1 | 2 (+ 2 at boundaries) |
| IFM rows in kernel window | K | K + 1 (one extra) |
| Local accumulators | 1 scalar | 4 scalars (ofm_0 … ofm_3) |
| Effective H-loop iterations | CONV_IN_HEIGHT | ≈ CONV_IN_HEIGHT / 2 |
| Theoretical latency speedup | 1× (baseline) | 2× |
| Empirical latency speedup | 1× (baseline) | ~1.9× |
| DSP utilisation | baseline | ~2–4× higher |
| LUT utilisation | baseline | higher (boundary guards) |
| BRAM utilisation | baseline | same |

The gap between theoretical 2× and empirical ~1.9× arises from boundary condition
evaluation overhead and the two-pass (forward + reverse) loop structure.

---

## 3. Loop tiling

Loop tiling is a parallelism technique that breaks large iteration
spaces into fixed-size tiles, enabling multiple channels to be processed simultaneously
and maximising on-chip data reuse.

Two tiling factors are defined, matching the Tm × Tn PE array in Fig. 5 of the
reference paper:

| Symbol | Description |
|--------|-------------|
| **Tm** | Number of output filters (OFM channels) processed per tile |
| **Tn** | Number of input IFM channels processed per tile |

For a layer with N total input channels and M total output filters:

```
Pn = floor(N / Tn)     number of full Tn-channel input tiles
Qn = N − Pn × Tn       remaining input channels      (0 ≤ Qn < Tn)

Pm = floor(M / Tm)     number of full Tm-channel output tiles
Qm = M − Pm × Tm       remaining output channels     (0 ≤ Qm < Tm)
```

The PE array instantiates Tm × Tn processing elements. PE(i, j) processes filter i,
channel j. Partial results from all Tn PEs in one row are summed to produce one partial
OFM channel. The wrapper iterates over Pm full output tiles and handles the Qm
remainder separately. Within each output tile the Pn full input tiles are processed by
`ring_1` / `parallel_1`, and the Qn remainder by `ring_2` / `parallel_2`.

Choosing (Tn, Tm) is a design-space tradeoff: larger tiles expose more parallelism but
consume more BRAM, DSP, and LUT. Different pairs are explored per layer (see Section 8).

---

## 4. Code architecture

### 4.1 RSD function hierarchy

```
tile_ring_conv2d()                     top-level tiled wrapper
├── ring_1_conv2d()                    Tn-channel tile, no padding
│     IFM [Tn][Ri][Ci]
│     F   [1][Tn][K][K]
│     OFM [1][Ro][Co]
│     Single scalar accumulator per (C, ks) pair
├── ring_2_conv2d()                    Qn-channel remainder, no padding
│     IFM [Qn][Ri][Ci]
│     F   [1][Qn][K][K]
│     OFM [1][Ro][Co]
├── ring_1_pad_conv2d()                Tn-channel tile, padding = 1
└── ring_2_pad_conv2d()                Qn-channel remainder, padding = 1
```

No-padding variants are used for LeNet-5 (output spatial dims shrink per layer).
Padded variants are used for all Mini VGGNet-32 layers (padding = 1, K = 3, stride = 1,
so output dims equal input dims).

### 4.2 PRSD function hierarchy

```
tile_parallel_conv2d()                 top-level tiled wrapper
├── parallel_1_conv2d()                Tn-channel tile, dual-row output
│     IFM [Tn][Ri][Ci]
│     F   [1][Tn][K][K]
│     OFM [1][Ro][Co]
│     Four local accumulators; writes 2 OFM rows per H pass
└── parallel_2_conv2d()                Qn-channel remainder, dual-row output
      IFM [Qn][Ri][Ci]
      F   [1][Qn][K][K]
      OFM [1][Ro][Co]
```

`parallel_1_conv2d` is a direct replacement for `ring_1_conv2d`, and
`parallel_2_conv2d` replaces `ring_2_conv2d`. Array interface shapes and the
top-level calling convention are identical, so cycle counts are directly comparable.

### 4.3 Top-level calling pattern

Both `tile_ring_conv2d` and `tile_parallel_conv2d` follow this structure:

```cpp
// --- Pm full output tiles ---
for (int i = 0; i < Pm; i++) {
    for (int tm = i*Tm; tm < (i+1)*Tm; tm++) {
        // Pn full input tiles
        for (int p = 0; p < Pn; p++) {
            // copy partial_IFM[Tn][Ri][Ci] and partial_F[1][Tn][K][K]
            ring_1_conv2d(partial_IFM, partial_F, partial_OFM);     // RSD
         // parallel_1_conv2d(partial_IFM, partial_F, partial_OFM); // PRSD
            // accumulate partial_OFM into OFM[tm]
        }
        if (Qn != 0) {
            // copy extra_IFM[Qn][Ri][Ci] and extra_F[1][Qn][K][K]
            ring_2_conv2d(extra_IFM, extra_F, extra_OFM);     // RSD
         // parallel_2_conv2d(extra_IFM, extra_F, extra_OFM); // PRSD
            // accumulate extra_OFM into OFM[tm]
        }
    }
}
// --- Qm output remainder ---
for (int qm = 0; qm < Qm; qm++) {
    // identical inner structure, writing to OFM[Pm*Tm + qm]
}
```

---

## 5. HLS pragma strategy

| Pragma | Applied to | Purpose |
|--------|------------|---------|
| `ARRAY_PARTITION complete dim=1` | IFM, F, OFM — channel dim | All channels accessible in one cycle; eliminates memory-port bottleneck |
| `ARRAY_PARTITION complete dim=2,3` | F — spatial (kernel) dims | All K×K weights readable simultaneously |
| `ARRAY_PARTITION complete` | partial_IFM, partial_F, partial_OFM | Full parallel access inside sub-functions |
| `PIPELINE II=1 rewind` | outer H loops | Single-cycle iteration interval; `rewind` removes inter-loop dead cycles |
| `UNROLL factor=15` | W loop (PRSD) | 15 parallel MAC chains per H step |
| `UNROLL factor=3` | i, j loops (K=3) | Fully unroll 3×3 kernel; all 9 products computed simultaneously |
| `UNROLL complete` | ks channel loop | All Tn multiply-accumulates execute in parallel |
| `bind_op op=fmul impl=dsp` | all multiplications | Map floating-point multiply to DSP48 slices |
| `bind_op op=fadd impl=dsp` | all accumulators and final adds | Map floating-point add/accumulate to DSP48 slices |
| `LOOP_TRIPCOUNT min= max=` | H, W loops | Guide the HLS latency estimator |
| `INTERFACE m_axi bundle=gmemN` | IFM, F, OFM ports | Separate AXI4 memory buses for concurrent data transfers |
| `INTERFACE s_axilite` | return port | AXI-Lite control interface for host |

---

## 6. CNN model architectures tested

### 6.1 LeNet-5

| Layer | Type | Output size | Filter / stride / padding | Tiling (Tn / Tm) |
|-------|------|-------------|---------------------------|------------------|
| INPUT | — | 32 × 32 × 1 | — | — |
| **CONV1** | Conv | 28 × 28 × 1 | 5×5, N=1, M=1, s=1, p=0 | **Tn=1 / Tm=1** |
| ACT | ReLU | 28 × 28 × 1 | — | — |
| POOL | MaxPool | 14 × 14 × 1 | 2×2 | — |
| **CONV2** | Conv | 10 × 10 × 16 | 5×5, N=6, M=16, s=1, p=0 | **Tn=2 / Tm=8** |
| ACT | ReLU | 10 × 10 × 16 | — | — |
| POOL | MaxPool | 5 × 5 × 16 | 2×2 | — |
| FC1 | Fully connected | 120 | — | — |
| FC2 | Fully connected | 84 | — | — |
| FC3 + Softmax | Output | 10 | — | — |

> LeNet-5 CONV1 is single-channel (N=1, M=1). The ~1.9× speedup is a spatial scan
> improvement and applies regardless of channel count.

### 6.2 Mini VGGNet-32

| Layer | Type | Output size | Filter / stride / padding | Tiling (Tn / Tm) |
|-------|------|-------------|---------------------------|------------------|
| INPUT | — | 32 × 32 × 3 | — | — |
| **CONV1** | Conv | 32 × 32 × 32 | 3×3, N=3, M=32, s=1, p=1 | **Tn=2 / Tm=16** |
| ACT | ReLU | 32 × 32 × 32 | — | — |
| BN | BatchNorm | 32 × 32 × 32 | — | — |
| **CONV2** | Conv | 32 × 32 × 32 | 3×3, N=32, M=32, s=1, p=1 | **Tn=8 / Tm=8** |
| ACT | ReLU | 32 × 32 × 32 | — | — |
| BN | BatchNorm | 32 × 32 × 32 | — | — |
| POOL | MaxPool | 16 × 16 × 32 | 2×2 | — |
| DROPOUT | — | 16 × 16 × 32 | — | — |
| **CONV3** | Conv | 16 × 16 × 64 | 3×3, N=64, M=64, s=1, p=1 | **Tn=32 / Tm=32** |
| ACT | ReLU | 16 × 16 × 64 | — | — |
| BN | BatchNorm | 16 × 16 × 64 | — | — |
| **CONV4** | Conv | 16 × 16 × 64 | 3×3, N=64, M=64, s=1, p=1 | **Tn=32 / Tm=64** |
| ACT | ReLU | 16 × 16 × 64 | — | — |
| BN | BatchNorm | 16 × 16 × 64 | — | — |
| POOL | MaxPool | 8 × 8 × 64 | 2×2 | — |
| DROPOUT | — | 8 × 8 × 64 | — | — |
| FC1 | Fully connected | 512 | — | — |
| ACT + BN | — | 512 | — | — |
| DROPOUT | — | 512 | — | — |
| FC2 + Softmax | Output | 10 | — | — |

> Only the four convolution layers (bold) are compared between RSD and PRSD.
> All other layers have identical cycle counts for both dataflows.

---

## 7. Performance model

### 7.1 Tile operation count formula

**Single-tile partial convolution operation counts:**

```
C1 = 4 × K² × Tn × W_out × H_out      full Tn-channel input tile
C2 = 4 × K² × Qn × W_out × H_out      Qn-channel input remainder
```

The factor 4 = 2 ops per MAC (1 multiply + 1 add) × 2 spatial kernel dimensions.

**Total operations for the full tiled convolution layer:**

```
TotalOps = 2 × [ Pm × Tm × (Pn × C1 + C2)  +  Qm × (Pn × C1 + C2) ]
```

Breaking this down:
- `Pn × C1 + C2` = ops to process all N input channels for one OFM channel
- `Tm × (...)` = repeat for Tm output channels per tile
- `Pm × ...` = repeat for all Pm output tiles
- `+ Qm × (...)` = Qm output-channel remainder
- outer `× 2` = forward + reverse H-loop passes

**TotalOps is identical for RSD and PRSD. Only the number of clock cycles differs.**

### 7.2 GOPS formula

```
GOPS = TotalOps / ( Cycles × T_clk × 10⁹ )

T_clk = 12.5 ns   (for simulation and synthesis in Vivado HLS, we took target clock = 80 MHz)
```

### 7.3 PRSD vs RSD speedup derivation

Since TotalOps cancels:

```
GOPS_PRSD     TotalOps / (C_PRSD × T_clk)     C_RSD
─────────── = ──────────────────────────────── = ───────  ≈  1.9
GOPS_RSD      TotalOps / (C_RSD  × T_clk)      C_PRSD
```

PRSD approximately halves the H-loop iteration count by producing two OFM rows per
pass. The theoretical maximum speedup is 2×. In practice it is ~1.9× due to boundary
condition evaluation overhead and the two-pass structure.

### 7.4 Worked examples — all layers

---

**LeNet-5 CONV1** — N=1, M=1, K=5, W_out=28, H_out=28, Tn=1, Tm=1

```
Pn = floor(1/1) = 1    Qn = 1 − 1×1 = 0
Pm = floor(1/1) = 1    Qm = 1 − 1×1 = 0

C1 = 4 × 25 × 1 × 28 × 28 = 78,400
C2 = 0  (Qn = 0)

TotalOps = 2 × [1 × 1 × (1 × 78,400 + 0) + 0] = 156,800

GOPS_RSD  = 156,800 / (65,891  × 12.5) = 0.1905
GOPS_PRSD = 156,800 / (35,136  × 12.5) = 0.3571
Speedup   = 65,891  / 35,136           = 1.876×
```

---

**LeNet-5 CONV2** — N=6, M=16, K=5, W_out=10, H_out=10, Tn=2, Tm=8

```
Pn = floor(6/2) = 3    Qn = 6 − 3×2 = 0
Pm = floor(16/8) = 2   Qm = 16 − 2×8 = 0

C1 = 4 × 25 × 2 × 10 × 10 = 20,000
C2 = 0  (Qn = 0)

TotalOps = 2 × [2 × 8 × (3 × 20,000 + 0) + 0]
         = 2 × [16 × 60,000] = 1,920,000

GOPS_RSD  = 1,920,000 / (89,080  × 12.5) = 1.7232
GOPS_PRSD = 1,920,000 / (46,214  × 12.5) = 3.3233
Speedup   = 89,080    / 46,214            = 1.927×
```

---

**VGGNet CONV1** — N=3, M=32, K=3, W_out=32, H_out=32, Tn=2, Tm=16

```
Pn = floor(3/2) = 1    Qn = 3 − 1×2 = 1
Pm = floor(32/16) = 2  Qm = 32 − 2×16 = 0

C1 = 4 × 9 × 2 × 32 × 32 = 73,728
C2 = 4 × 9 × 1 × 32 × 32 = 36,864

TotalOps = 2 × [2 × 16 × (1 × 73,728 + 36,864) + 0]
         = 2 × [32 × 110,592] = 7,077,888

GOPS_RSD  = 7,077,888 / (114,088 × 12.5) = 4.9620
GOPS_PRSD = 7,077,888 / (60,872  × 12.5) = 9.3010
Speedup   = 114,088   / 60,872            = 1.875×
```

---

**VGGNet CONV2** — N=32, M=32, K=3, W_out=32, H_out=32, Tn=8, Tm=8

```
Pn = floor(32/8) = 4   Qn = 32 − 4×8 = 0
Pm = floor(32/8) = 4   Qm = 32 − 4×8 = 0

C1 = 4 × 9 × 8 × 32 × 32 = 294,912
C2 = 0  (Qn = 0)

TotalOps = 2 × [4 × 8 × (4 × 294,912) + 0]
         = 2 × [32 × 1,179,648] = 75,497,472

GOPS_RSD  = 75,497,472 / (833,672  × 12.5) = 7.2290
GOPS_PRSD = 75,497,472 / (433,443  × 12.5) = 13.9010
Speedup   = 833,672    / 433,443            = 1.923×
```

---

**VGGNet CONV3** — N=64, M=64, K=3, W_out=16, H_out=16, Tn=32, Tm=32

```
Pn = floor(64/32) = 2  Qn = 64 − 2×32 = 0
Pm = floor(64/32) = 2  Qm = 64 − 2×32 = 0

C1 = 4 × 9 × 32 × 16 × 16 = 294,912
C2 = 0  (Qn = 0)

TotalOps = 2 × [2 × 32 × (2 × 294,912) + 0]
         = 2 × [64 × 589,824] = 75,497,472

GOPS_RSD  = 75,497,472 / (4,810,752 × 12.5) = 1.2530
GOPS_PRSD = 75,497,472 / (2,565,187 × 12.5) = 2.3490
Speedup   = 4,810,752  / 2,565,187           = 1.875×
```

---

**VGGNet CONV4** — N=64, M=64, K=3, W_out=16, H_out=16, Tn=32, Tm=64

```
Pn = floor(64/32) = 2  Qn = 64 − 2×32 = 0
Pm = floor(64/64) = 1  Qm = 64 − 1×64 = 0

C1 = 4 × 9 × 32 × 16 × 16 = 294,912
C2 = 0  (Qn = 0)

TotalOps = 2 × [1 × 64 × (2 × 294,912) + 0]
         = 2 × [64 × 589,824] = 75,497,472

GOPS_RSD  = 75,497,472 / (4,810,752 × 12.5) = 1.2530
GOPS_PRSD = 75,497,472 / (2,565,187 × 12.5) = 2.3490
Speedup   = 4,810,752  / 2,565,187           = 1.875×
```

---
## 8. Verified results — TABLE II

Clock: 80 MHz (T_clk = 12.5 ns). Cycle counts and GOPS values from hardware
simulation. Speedup = C_RSD / C_PRSD.

### 8.1 LeNet-5

| Parameter | CONV1 (1st) | CONV2 (2nd) |
|-----------|-------------|-------------|
| IFM | 32 × 32 × 1 | 14 × 14 × 6 |
| Kernel (K×K×N×M) | 5×5×1×6 | 5×5×6×16 |
| OFM | 28 × 28 × 1 | 10 × 10 × 16 |
| Tm / Tn | 1 / 1 | 8 / 2 |
| Cycles — RSD | 65,891 | 89,080 |
| Cycles — PRSD | 35,136 | 46,214 |
| **GOPS — RSD** | **0.2068** | **82.90** |
| **GOPS — PRSD** | **0.3878** | **159.8** |
| **Speedup (C_RSD / C_PRSD)** | **1.876×** | **1.927×** |

### 8.2 Mini VGGNet-32

| Parameter | CONV1 (1st) | CONV2 (2nd) | CONV3 (3rd) | CONV4 (4th) |
|-----------|-------------|-------------|-------------|-------------|
| IFM | 32×32×3 | 32×32×32 | 16×16×64 | 16×16×64 |
| Kernel (K×K×N×M) | 3×3×3×32 | 3×3×32×32 | 3×3×64×64 | 3×3×64×64 |
| OFM | 32×32×32 | 32×32×32 | 16×16×64 | 16×16×64 |
| Tm / Tn | 16 / 2 | 8 / 8 | 32 / 32 | 64 / 32 |
| Pn / Qn | 1 / 1 | 4 / 0 | 2 / 0 | 2 / 0 |
| Pm / Qm | 2 / 0 | 4 / 0 | 2 / 0 | 1 / 0 |
| C1 | 73,728 | 294,912 | 294,912 | 294,912 |
| C2 | 36,864 | 0 | 0 | 0 |
| **TotalOps** | **7,077,888** | **75,497,472** | **75,497,472** | **75,497,472** |
| Cycles — RSD | 114,088 | 833,672 | 4,810,752 | 4,810,752 |
| Cycles — PRSD | 60,872 | 433,443 | 2,565,187 | 2,565,187 |
| **GOPS — RSD** | **4.96** | **1.81** | **0.63** | **0.63** |
| **GOPS — PRSD** | **9.3** | **3.48** | **1.18** | **1.18** |
| **Speedup (C_RSD / C_PRSD)** | **1.875×** | **1.923×** | **1.875×** | **1.875×** |

> **Key observation:** The GOPS ratio equals the cycle ratio in every row because
> TotalOps is mathematically same for RSD and PRSD. The consistent 1.875–1.927× range across
> six layers and two models confirms the PRSD design objective of approximately 1.9×
> lower latency.


---

## 9. Hardware utilisation tradeoff

PRSD achieves its latency reduction by maintaining four local floating-point
accumulators in parallel and evaluating additional spatial boundary conditions per
pixel. This increases resource consumption as follows:

| Resource | RSD | PRSD | Cause of increase |
|----------|-----|------|-------------------|
| DSP48 | moderate | ~2–4× higher | 4 accumulators × fmul + fadd; W-loop unrolled ×15 |
| LUT | moderate | higher | 5 extra boolean boundary guards per pixel |
| BRAM | same | same | Array dimensions unchanged |
| Flip-flop | moderate | higher | Deeper accumulation trees need more pipeline registers |

Exact figures depend on the target device, Tn, and Tm. This can be checked from  the synthesis report after running HLS, can refer the screenshot section for this.

**Design guidance:**
- PRSD is most beneficial in latency-critical deployments where extra DSP and LUT
  resources are available (e.g. larger Ultrascale+ devices).
- For area-constrained deployments, RSD with larger tiling factors may achieve better
  efficiency per unit area.
- The speedup is independent of Tn, Tm, K, N, M — it depends only on the spatial
  dimensions and the two-row scanning pattern.

---

## 10. File structure

```
.
├── README.md
│
├── rsd/
│   ├── ring_1_conv2d.cpp          RSD: Tn-channel tile, no padding
│   ├── ring_2_conv2d.cpp          RSD: Qn-channel remainder, no padding
│   ├── ring_1_pad_conv2d.cpp      RSD: Tn-channel tile, padding = 1
│   ├── ring_2_pad_conv2d.cpp      RSD: Qn-channel remainder, padding = 1
│   └── ring_conv2d.h
│
├── prsd/
│   ├── parallel_1_conv2d.cpp      PRSD: Tn-channel tile, dual-row output
│   ├── parallel_2_conv2d.cpp      PRSD: Qn-channel remainder, dual-row output
│   └── parallel_conv2d.h
│
├── tiled/
│   ├── tile_ring_conv2d.cpp       RSD top-level tiled wrapper
│   ├── tile_parallel_conv2d.cpp   PRSD top-level tiled wrapper
│   └── tile_conv2d.h              Shared tiling parameters (Tn, Tm, Pn, Pm …)
│
├── tb/
│   ├── tb_rsd_lenet.cpp
│   ├── tb_rsd_vgg.cpp
│   ├── tb_prsd_lenet.cpp
│   └── tb_prsd_vgg.cpp
│
└── scripts/
    └── run_hls.tcl
```

---



## 11. References

1. *ESSA: An Energy-Aware Bit-Serial Streaming Deep Convolutional Neural Network
   Accelerator*: primary reference for RSD, loop tiling definition, PE array
   architecture (Fig. 5), and the TABLE II cycle-count benchmarks used in Section 8.

2. *Optimizing FPGA-based Accelerator Design for Deep
   Convolutional Neural Networks*: major references for loop unrolling, tile sizing
   for loop tiling concepts.
