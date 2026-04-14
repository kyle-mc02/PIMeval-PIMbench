# Sparse Convolution
A sparse convolution operator designed for event-based camera data, based on the three-stage pipeline proposed by Zhang et al. (2025). Event-based cameras are bio-inspired vision sensors that only report asynchronous pixel-level brightness changes, producing highly sparse output where typically 95-99% of pixels are zero. Traditional dense convolution wastes significant computation processing these zero-value regions. The sparse convolution operator addresses this by: (1) detecting valid sub-convolutions using the locations of active pixels rather than exhaustively scanning all sliding windows, (2) gathering only the valid image patches into a compact dense matrix via sparse im2col, eliminating zero-computation columns, and (3) performing standard GEMM on the reduced matrix. This approach reduces computational workload by up to 90% while producing results identical to dense convolution.

## Directory Structure
```
sparse-convolution/
├── PIM/
│   ├── Makefile
│   ├── sparse-conv.cpp
├── baselines/
│   ├── sparse-conv.py
├── README.md
├── Makefile
```

## Implementation Description
This repository contains two different implementations of the sparse convolution benchmark:
1. CPU & GPU
2. PIM

### Baseline Implementation
CPU and GPU have been used as baselines.

#### CPU & GPU
* The CPU and GPU variants of sparse convolution have been implemented using standard Python and PyTorch framework for machine learning. The same script is used for both, the device is specified from the command line by the user. The script implements the full three-stage sparse convolution pipeline: valid sub-convolution detection, sparse im2col, and GEMM. It generates synthetic sparse input data simulating event-based camera output, where a configurable fraction of pixels are set to zero. Both the sparse pipeline and a standard dense convolution (via PyTorch nn.Conv2d) are executed for timing comparison.
* GPU and CPU by default have a batch size of 1, and the batch size can be specified differently for CPU and GPU from the command line.
* The number of valid sub-convolutions, total execution time for both sparse and dense paths in ms, and the speedup ratio are printed at the end. Command-line arguments specify the device, and cpu is chosen as the default device. If the device is specified as cuda, and cuda is not available, cpu is chosen as the fallback option.

### PIM Implementation
The PIM variant is implemented using C++. The C++ implementation includes the full three-stage CPU sparse convolution pipeline for baseline profiling and verification.

## Compilation and Execution Instructions for Specific Variants

### CPU Variant
To run the script for the CPU variant, use command like the following example:
```bash
cd baselines
python3 sparse-conv.py -b 1 -d 3 -r 224 -c 224 -kr 3 -kc 3 -kd 64 -s 1 -p 1 -sp 0.95
```
Note:
 * "-b" to specify the batch size for the input.
 * "-d" to specify the depth (channels) of the input matrix.
 * "-r" to specify the number of rows (height) in the input matrix.
 * "-c" to specify the number of columns (width) in the input matrix.
 * "-kr" to specify the number of rows in the kernel matrix.
 * "-kc" to specify the number of columns in the kernel matrix.
 * "-kd" to specify the depth of the kernel matrix (number of output channels).
 * "-s" to specify the stride.
 * "-p" to specify the input padding.
 * "-sp" to specify the sparsity level as a fraction (0.0-1.0). Default -> 0.95.

### GPU Variant
To run the script for the GPU variant, use command like the following example:
```bash
cd baselines
python3 sparse-conv.py -cuda
```
Note:
 * "-cuda" is specified to use GPU for the sparse convolution. Default -> CPU.
 * For GPU, it is assumed that the system has a GPU with CUDA support.

### PIM Variant
To compile for the PIM variant, use:
```bash
cd PIM
make
```

## Execution Instructions

### Running the Executable
After compiling, run the executable like the following example command:
```bash
./sparse-conv.out -r 224 -c 224 -d 3 -l 3 -w 3 -z 64 -p 1 -s 1 -a 95 -v t
```
Note:
 * "-d" to specify the depth (channels) of the input matrix.
 * "-r" to specify the number of rows in the input matrix.
 * "-c" to specify the number of columns in the input matrix.
 * "-l" to specify the kernel height.
 * "-w" to specify the kernel width.
 * "-z" to specify the depth of the kernel matrix (number of output channels).
 * "-s" to specify the stride.
 * "-p" to specify the input padding.
 * "-a" to specify the sparsity as a percentage of zeros (0-100). Default -> 95.
 * "-v t" to compare the PIM results with CPU results and print the mismatches. Default -> not compared.
 * "-m t" to enable additional debug prints. Default -> disabled.
 * "-o" to specify a DRAM configuration file path.

## References
* Zhang Sen et al., "High-efficiency sparse convolution operator for event-based cameras", Frontiers in Neurorobotics, Volume 19, 2025, DOI: 10.3389/fnbot.2025.1537673
* F. A. Siddique et al., "Architectural Modeling and Benchmarking for Digital DRAM PIM", IEEE IISWC, 2024, DOI: 10.1109/IISWC63097.2024.00030
