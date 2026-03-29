// File: sparse_conv.cpp
// Sparse Convolution using PIMeval
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.
//
// This benchmark extends the dense convolution (PIMbench/convolution/PIM/conv.cpp)
// with two optional sparse optimizations that can be toggled independently:
//
//   Weight sparsity (-W): Skip pimScaledAdd calls for zero-valued kernel weights.
//     - Reduces PIM operation count proportional to kernel sparsity.
//     - Zero bookkeeping overhead; works on the host-side loop.
//
//   Input sparsity (-I): CSR-compress the im2col matrix to exclude output positions
//     whose entire receptive field (across all input channels) is zero.
//     - Reduces PIM object size, data transfer volume, and operation count.
//     - Requires precomputing a global active-position map before the main loop.
//     - Results are scattered back to the full output tensor after PIM.
//
// Both optimizations can be active simultaneously.
//
// Usage example:
//   ./sparse_conv.out -r 224 -c 224 -d 3 -z 64 -k 0.7 -a 0.5 -W -I -v t
//
//   -k 0.7   : inject 70% zeros into kernel weights
//   -a 0.5   : inject 50% zeros into input activations
//   -W       : enable weight sparsity optimization
//   -I       : enable input activation sparsity optimization

#include "libpimeval.h"
#include <cmath>
#include <getopt.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "util.h"
#include <chrono>
#include <iomanip>
#include <assert.h>

using namespace std;
typedef vector<vector<vector<int>>> Image3D;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

typedef struct Params {
  uint64_t row, column, dim, stride, kernelHeight, kernelWidth, kernelDim, padding;
  float    kernelSparsity;    // fraction of kernel weights forced to zero [0.0, 1.0]
  float    inputSparsity;     // fraction of input activations forced to zero [0.0, 1.0]
  bool     useWeightSparseOpt; // -W: skip pimScaledAdd for zero weights
  bool     useInputSparseOpt;  // -I: CSR-compress im2col to active output positions
  char    *kernelMatrixFile;
  char    *imageMatrixFile;
  char    *dramConfigFile;
  bool     shouldVerify;
  bool     moreDebugPrints;
} Params;

void usage()
{
  fprintf(stderr,
    "\nUsage:  ./sparse_conv.out [options]"
    "\n"
    "\n  Standard conv options:"
    "\n    -r    input rows          (default=224)"
    "\n    -c    input columns       (default=224)"
    "\n    -d    input depth/channels(default=3)"
    "\n    -s    stride              (default=1)"
    "\n    -l    kernel height       (default=3)"
    "\n    -w    kernel width        (default=3)"
    "\n    -z    number of kernels   (default=64)"
    "\n    -p    padding             (default=1)"
    "\n    -v    verify against CPU  (default=false, use -v t)"
    "\n    -f    kernel matrix file  (default=generated)"
    "\n    -i    image matrix file   (default=generated)"
    "\n    -o    DRAM config file    (default=functional sim)"
    "\n    -m    more debug prints   (default=false, use -m t)"
    "\n"
    "\n  Sparsity injection options (zero out a fraction of values):"
    "\n    -k    kernel weight sparsity ratio [0.0-1.0] (default=0.0)"
    "\n    -a    input activation sparsity ratio [0.0-1.0] (default=0.0)"
    "\n"
    "\n  Sparse optimization flags:"
    "\n    -W    enable weight sparsity optimization (skip zero-weight pimScaledAdd calls)"
    "\n    -I    enable input activation sparsity optimization (CSR-compress im2col matrix)"
    "\n"
    "\n  Example: ./sparse_conv.out -r 56 -c 56 -d 64 -z 64 -k 0.5 -a 0.3 -W -I -v t"
    "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row             = 224;
  p.column          = 224;
  p.dim             = 3;
  p.stride          = 1;
  p.kernelHeight    = 3;
  p.kernelWidth     = 3;
  p.kernelDim       = 64;
  p.padding         = 1;
  p.kernelSparsity  = 0.0f;
  p.inputSparsity   = 0.0f;
  p.useWeightSparseOpt = false;
  p.useInputSparseOpt  = false;
  p.kernelMatrixFile = nullptr;
  p.imageMatrixFile  = nullptr;
  p.dramConfigFile   = nullptr;
  p.shouldVerify     = false;
  p.moreDebugPrints  = false;

  int opt;
  while ((opt = getopt(argc, argv, "r:c:d:s:l:w:z:p:v:f:i:o:m:k:a:WI")) >= 0) {
    switch (opt) {
      case 'h': usage(); exit(0);
      case 'r': p.row           = atoi(optarg); break;
      case 'c': p.column        = atoi(optarg); break;
      case 'd': p.dim           = atoi(optarg); break;
      case 's': p.stride        = atoi(optarg); break;
      case 'l': p.kernelHeight  = atoi(optarg); break;
      case 'w': p.kernelWidth   = atoi(optarg); break;
      case 'z': p.kernelDim     = atoi(optarg); break;
      case 'p': p.padding       = atoi(optarg); break;
      case 'v': p.shouldVerify  = (*optarg == 't'); break;
      case 'f': p.kernelMatrixFile = optarg; break;
      case 'i': p.imageMatrixFile  = optarg; break;
      case 'o': p.dramConfigFile   = optarg; break;
      case 'm': p.moreDebugPrints  = (*optarg == 't'); break;
      case 'k': p.kernelSparsity   = atof(optarg); break;
      case 'a': p.inputSparsity    = atof(optarg); break;
      case 'W': p.useWeightSparseOpt = true; break;
      case 'I': p.useInputSparseOpt  = true; break;
      default:
        fprintf(stderr, "\nUnrecognized option!\n");
        usage();
        exit(0);
    }
  }
  return p;
}

// ---------------------------------------------------------------------------
// Sparsity injection
// ---------------------------------------------------------------------------

// Zero out approximately `sparsity` fraction of the non-padding elements of a
// matrix that was produced by getMatrix() (which already has padding borders).
void applySparsity(float sparsity, int padding, vector<vector<int>> &mat)
{
  if (sparsity <= 0.0f) return;
  static mt19937 rng(42);
  uniform_real_distribution<float> dist(0.0f, 1.0f);
  int rows = mat.size();
  int cols = (rows > 0) ? mat[0].size() : 0;
  for (int i = padding; i < rows - padding; i++) {
    for (int j = padding; j < cols - padding; j++) {
      if (dist(rng) < sparsity) {
        mat[i][j] = 0;
      }
    }
  }
}

// Zero out approximately `sparsity` fraction of all kernel elements (no padding).
void applyKernelSparsity(float sparsity, vector<vector<int>> &mat)
{
  if (sparsity <= 0.0f) return;
  static mt19937 rng(99);
  uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &row : mat) {
    for (auto &val : row) {
      if (dist(rng) < sparsity) val = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// im2col: dense decomposition (identical to conv.cpp)
// ---------------------------------------------------------------------------

// Decomposes a 2D input channel into a matrix suitable for convolution.
// decompMatrix[filter_pos][output_pos] = input value at that receptive field.
// Rows = KH*KW filter positions. Cols = outH*outW output positions.
void getDecomposedMatrix(int matrixRow, int matrixColumn,
                         int filterRow, int filterColumn, int stride,
                         vector<vector<int>> &inputMatrix,
                         vector<vector<int>> &decompMatrix)
{
  decompMatrix.resize(filterRow * filterColumn,
                      vector<int>(matrixRow * matrixColumn, 0));
  int colIdx = 0;
  for (uint64_t i = 0; i < (inputMatrix.size() - filterRow + 1); i += stride) {
    for (uint64_t j = 0; j < (inputMatrix[i].size() - filterColumn + 1); j += stride) {
      int rowIDX = 0;
      for (uint64_t k = i; k < i + (uint64_t)filterRow; k++) {
        for (uint64_t l = j; l < j + (uint64_t)filterColumn; l++) {
          decompMatrix[rowIDX++][colIdx] = inputMatrix[k][l];
        }
      }
      ++colIdx;
    }
  }
}

// ---------------------------------------------------------------------------
// Active output position computation (for input sparsity optimization)
// ---------------------------------------------------------------------------

// Compute which flat output positions [0, outH*outW) have at least one non-zero
// value in their receptive field across any input channel. Only these positions
// can produce a non-zero convolution output regardless of kernel values.
//
// inputMatrix[d] is the padded input for channel d (size = (H+2p) x (W+2p)).
// outW is derived from the actual inputMatrix dimensions.
void computeActiveOutputPositions(
    const vector<vector<vector<int>>> &inputMatrix,  // [depth][paddedH][paddedW]
    int filterRow, int filterColumn, int stride,
    int outH, int outW,
    vector<int> &activePositions)
{
  int depth = inputMatrix.size();
  vector<bool> active(outH * outW, false);

  for (int d = 0; d < depth; d++) {
    for (int r = 0; r < outH; r++) {
      for (int c = 0; c < outW; c++) {
        int idx = r * outW + c;
        if (active[idx]) continue;
        for (int kh = 0; kh < filterRow && !active[idx]; kh++) {
          for (int kw = 0; kw < filterColumn && !active[idx]; kw++) {
            if (inputMatrix[d][r * stride + kh][c * stride + kw] != 0) {
              active[idx] = true;
            }
          }
        }
      }
    }
  }

  activePositions.clear();
  for (int i = 0; i < outH * outW; i++) {
    if (active[i]) activePositions.push_back(i);
  }
}

// ---------------------------------------------------------------------------
// Compressed im2col (for input sparsity optimization)
// ---------------------------------------------------------------------------

// Like getDecomposedMatrix but only produces columns for the output positions
// in activePositions. activePositions contains flat indices into [0, outH*outW).
// Output: compDecomp[filter_pos][compressed_col_idx].
void getCompressedDecompMatrix(int matrixRow, int matrixColumn,
                               int filterRow, int filterColumn, int stride,
                               vector<vector<int>> &inputMatrix,
                               const vector<int> &activePositions,
                               vector<vector<int>> &compDecomp)
{
  // Build full decomp first then select the active columns.
  // This is correct for the simulator (host memory is not the bottleneck here).
  vector<vector<int>> fullDecomp;
  getDecomposedMatrix(matrixRow, matrixColumn, filterRow, filterColumn, stride,
                      inputMatrix, fullDecomp);

  int numFilterPos = filterRow * filterColumn;
  int numActive    = activePositions.size();
  compDecomp.assign(numFilterPos, vector<int>(numActive, 0));
  for (int ci = 0; ci < numActive; ci++) {
    int origCol = activePositions[ci];
    for (int row = 0; row < numFilterPos; row++) {
      compDecomp[row][ci] = fullDecomp[row][origCol];
    }
  }
}

// ---------------------------------------------------------------------------
// PIM convolution execution
// ---------------------------------------------------------------------------

// Runs one convolution pass on the PIM device.
//
// filterMatrix : the 2D kernel [KH][KW]
// inputMatrix  : the im2col decomposed (and possibly compressed) matrix
//                [KH*KW rows x N output positions cols]
// outputVector : in/out running accumulator (initialized by caller to zeros
//                or a previous partial sum)
// numRequiredPIMRows : KH*KW (number of filter positions = rows in inputMatrix)
// numRequiredPIMCol  : N output positions (= inputMatrix[0].size())
// skipZeroWeights    : if true, skip pimScaledAdd for zero-valued kernel weights
//
// After return, outputVector is resized to numRequiredPIMCol and holds the
// updated partial sums.
void performConv(vector<vector<int>> &filterMatrix,
                 vector<vector<int>> &inputMatrix,
                 vector<int>         &outputVector,
                 uint64_t             numRequiredPIMRows,
                 int                  numRequiredPIMCol,
                 bool                 skipZeroWeights,
                 bool                 moreDebugPrints)
{
  if (filterMatrix.empty() || inputMatrix.empty()) return;

  PimObjId filterObject = pimAlloc(PIM_ALLOC_AUTO, numRequiredPIMCol, PIM_INT32);
  if (filterObject == -1) {
    cerr << "Abort: pimAlloc failed for filterObject" << endl;
    return;
  }

  // Initialize the accumulator with the current running total.
  PimStatus status = pimCopyHostToDevice((void *)outputVector.data(), filterObject);
  if (status != PIM_OK) {
    cerr << "Abort: pimCopyHostToDevice failed for filterObject" << endl;
    return;
  }

  PimObjId matrixObject = pimAllocAssociated(filterObject, PIM_INT32);
  if (matrixObject == -1) {
    cerr << "Abort: pimAllocAssociated failed for matrixObject" << endl;
    return;
  }

  int col = filterMatrix[0].size(); // = KW

  for (uint64_t j = 0; j < inputMatrix.size(); j += numRequiredPIMRows) {
    for (uint64_t i = 0; i < numRequiredPIMRows; i++) {
      int weight = filterMatrix[i / col][i % col];

      // Weight sparsity optimization: skip this filter position entirely.
      // This saves both the host-to-device transfer and the PIM operation.
      if (skipZeroWeights && weight == 0) {
        if (moreDebugPrints) {
          cout << "  [WeightSparse] skipping filter pos " << i
               << " (weight=0)" << endl;
        }
        continue;
      }

      status = pimCopyHostToDevice((void *)inputMatrix[i + j].data(), matrixObject);
      if (status != PIM_OK) {
        cerr << "Abort: pimCopyHostToDevice failed for matrixObject at row " << i << endl;
        return;
      }

      // filterObject += matrixObject * weight
      status = pimScaledAdd(matrixObject, filterObject, filterObject, weight);
      if (status != PIM_OK) {
        cerr << "Abort: pimScaledAdd failed at row " << i << endl;
        return;
      }
    }
  }

  outputVector.resize(numRequiredPIMCol);
  status = pimCopyDeviceToHost(filterObject, (void *)outputVector.data());
  if (status != PIM_OK) {
    cerr << "Abort: pimCopyDeviceToHost failed" << endl;
    return;
  }

  pimFree(filterObject);
  pimFree(matrixObject);
}

// ---------------------------------------------------------------------------
// Tree reduction across channel chunks (identical to conv.cpp)
// ---------------------------------------------------------------------------

// Reduces inputVector (which contains numChunks blocks of hopSize elements)
// into outputVector (of hopSize elements) by summing across blocks using
// a binary tree of PIM add operations.
void aggregate(vector<int> &inputVector, vector<int> &outputVector, unsigned hopSize)
{
  uint64_t numChunks  = inputVector.size() / hopSize;
  uint64_t remChunk   = numChunks;

  while (remChunk > 1) {
    uint64_t reduceChunks = remChunk;
    vector<int> tempVector(hopSize, 0);

    // If odd number of chunks, save the last one and exclude from this round.
    if (remChunk % 2) {
      copy(inputVector.end() - hopSize, inputVector.end(), tempVector.begin());
      reduceChunks = remChunk - 1;
    }

    uint64_t length = (reduceChunks / 2) * hopSize;
    PimObjId srcObj = pimAlloc(PIM_ALLOC_AUTO, length, PIM_INT32);
    PimObjId dstObj = pimAllocAssociated(srcObj, PIM_INT32);
    if (srcObj == -1 || dstObj == -1) {
      cerr << "Abort: pimAlloc failed in aggregate" << endl;
      return;
    }

    pimCopyHostToDevice((void *)inputVector.data(), srcObj);                    // left halves
    pimCopyHostToDevice((void *)(inputVector.data() + length), dstObj);         // right halves
    pimAdd(srcObj, dstObj, dstObj);
    inputVector.resize(length);
    pimCopyDeviceToHost(dstObj, (void *)inputVector.data());
    pimFree(srcObj);
    pimFree(dstObj);

    // Re-add the saved odd chunk if needed.
    if (reduceChunks != remChunk) {
      PimObjId finalSrc = pimAlloc(PIM_ALLOC_AUTO, hopSize, PIM_INT32);
      PimObjId finalDst = pimAllocAssociated(finalSrc, PIM_INT32);
      if (finalSrc == -1 || finalDst == -1) {
        cerr << "Abort: pimAlloc failed in aggregate (odd-chunk merge)" << endl;
        return;
      }
      pimCopyHostToDevice((void *)inputVector.data(), finalSrc);
      pimCopyHostToDevice((void *)tempVector.data(), finalDst);
      pimAdd(finalSrc, finalDst, finalDst);
      pimCopyDeviceToHost(finalDst, (void *)inputVector.data());
      pimFree(finalSrc);
      pimFree(finalDst);
    }

    remChunk = reduceChunks / 2;
  }

  outputVector = inputVector;
}

// ---------------------------------------------------------------------------
// CPU reference implementation for verification
// ---------------------------------------------------------------------------

void VerifyWithCPU(Image3D &input, Image3D &kernel,
                   int padding, int stride, bool moreDebugPrints,
                   Image3D &PIMResult)
{
  int inputDepth   = input.size();
  int inputHeight  = input[0].size();
  int inputWidth   = input[0][0].size();
  int kernelDepth  = kernel.size();
  int kernelHeight = kernel[0].size();
  int kernelWidth  = kernel[0][0].size();
  int outputHeight = (inputHeight - kernelHeight) / stride + 1;
  int outputWidth  = (inputWidth  - kernelWidth)  / stride + 1;
  int outputDepth  = kernelDepth;

  if (outputHeight <= 0 || outputWidth <= 0 || outputDepth <= 0) {
    cerr << "Invalid output dimensions for CPU verification." << endl;
    exit(1);
  }

  Image3D output(outputDepth,
                 vector<vector<int>>(outputHeight, vector<int>(outputWidth, 0)));

  cout << "Performing convolution on CPU for verification..." << endl;
#pragma omp parallel for collapse(3)
  for (int k = 0; k < kernelDepth; ++k) {
    for (int i = 0; i < outputHeight; ++i) {
      for (int j = 0; j < outputWidth; ++j) {
        int convSum = 0;
        for (int d = 0; d < inputDepth; ++d) {
          for (int h = 0; h < kernelHeight; ++h) {
            for (int w = 0; w < kernelWidth; ++w) {
              convSum += kernel[k][h][w] * input[d][i * stride + h][j * stride + w];
            }
          }
        }
        output[k][i][j] = convSum;
      }
    }
  }

  int mismatches = 0;
  cout << "Comparing PIM results with CPU results..." << endl;
  for (uint64_t i = 0; i < output.size(); ++i) {
    for (uint64_t j = 0; j < output[0].size(); ++j) {
      for (uint64_t k = 0; k < output[0][0].size(); ++k) {
        if (output[i][j][k] != PIMResult[i][j][k]) {
          ++mismatches;
          if (moreDebugPrints) {
            cout << "  Mismatch at [" << i << "][" << j << "][" << k << "]:"
                 << " PIM=" << PIMResult[i][j][k]
                 << " CPU=" << output[i][j][k] << endl;
          }
        }
      }
    }
  }

  if (mismatches == 0) {
    cout << "Success: PIM results match CPU." << endl;
  } else {
    cout << "Failure: " << mismatches << " mismatches between PIM and CPU." << endl;
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);

  // -------------------------------------------------------------------------
  // Print configuration
  // -------------------------------------------------------------------------
  cout << "Sparse Convolution Configuration:" << endl;
  cout << "  Input:  " << params.row << "x" << params.column
       << "x" << params.dim << endl;
  cout << "  Kernel: " << params.kernelHeight << "x" << params.kernelWidth
       << "x" << params.kernelDim << "  stride=" << params.stride
       << "  padding=" << params.padding << endl;
  cout << "  Kernel sparsity injection:    " << params.kernelSparsity * 100.0f << "%" << endl;
  cout << "  Input sparsity injection:     " << params.inputSparsity  * 100.0f << "%" << endl;
  cout << "  Weight sparse optimization:   " << (params.useWeightSparseOpt ? "ON" : "OFF") << endl;
  cout << "  Input sparse optimization:    " << (params.useInputSparseOpt  ? "ON" : "OFF") << endl;

  // -------------------------------------------------------------------------
  // Generate input and kernel matrices
  // -------------------------------------------------------------------------
  Image3D inputMatrix(params.dim);
  for (uint64_t i = 0; i < params.dim; i++) {
    getMatrix(params.row, params.column, params.padding, inputMatrix[i]);
    applySparsity(params.inputSparsity, params.padding, inputMatrix[i]);
  }

  Image3D kernelMatrix(params.kernelDim);
  for (auto &mat : kernelMatrix) {
    getMatrix(params.kernelHeight, params.kernelWidth, 0, mat);
    applyKernelSparsity(params.kernelSparsity, mat);
  }

  // Report actual measured sparsity after injection
  {
    uint64_t totalKernelElements = 0, zeroKernelElements = 0;
    for (auto &mat : kernelMatrix) {
      for (auto &row : mat) {
        for (int v : row) { ++totalKernelElements; if (v == 0) ++zeroKernelElements; }
      }
    }
    uint64_t totalInputElements = 0, zeroInputElements = 0;
    for (auto &ch : inputMatrix) {
      for (auto &row : ch) {
        for (int v : row) { ++totalInputElements; if (v == 0) ++zeroInputElements; }
      }
    }
    cout << fixed << setprecision(1);
    cout << "  Actual kernel sparsity:       "
         << 100.0 * zeroKernelElements / totalKernelElements << "%" << endl;
    cout << "  Actual input sparsity:        "
         << 100.0 * zeroInputElements / totalInputElements << "%" << endl;
  }

  // -------------------------------------------------------------------------
  // Create PIM device
  // -------------------------------------------------------------------------
  if (!createDevice(params.dramConfigFile)) return 1;

  PimDeviceProperties deviceProp;
  if (pimGetDeviceProperties(&deviceProp) != PIM_OK) {
    cerr << "Abort: pimGetDeviceProperties failed" << endl;
    return 1;
  }

  uint64_t numCols     = deviceProp.numColPerSubarray;
  uint64_t numRows     = deviceProp.numRowPerSubarray;
  uint64_t numOfBits   = (uint64_t)deviceProp.numRanks *
                         (uint64_t)deviceProp.numBankPerRank *
                         (uint64_t)deviceProp.numSubarrayPerBank *
                         numCols * numRows;

  // -------------------------------------------------------------------------
  // Derived dimensions
  // -------------------------------------------------------------------------
  int inputDepth   = inputMatrix.size();
  int inputHeight  = inputMatrix[0].size();   // padded
  int inputWidth   = inputMatrix[0][0].size(); // padded
  int kernelH      = kernelMatrix[0].size();
  int kernelW      = kernelMatrix[0][0].size();

  int outMatDim = params.kernelDim;
  int outMatRow = (int)floor((inputHeight - kernelH) / (double)params.stride) + 1;
  int outMatCol = (int)floor((inputWidth  - kernelW) / (double)params.stride) + 1;
  int numOfPIMRow = params.kernelHeight * params.kernelWidth;  // = KH*KW

  // Max channels per chunk for the DENSE path (limited by PIM capacity)
  int numOfMatPerRow =
      (int64_t)floor((1.0 * numOfBits) / (outMatRow * outMatCol)) < (int64_t)params.dim
      ? (int)floor((1.0 * numOfBits) / (outMatRow * outMatCol))
      : (int)params.dim;
  if (numOfMatPerRow < 1) numOfMatPerRow = 1;

  // -------------------------------------------------------------------------
  // Input sparsity optimization: compute active output positions
  // -------------------------------------------------------------------------
  // activePositions[i] = flat output position index (in [0, outH*outW)) that
  // has at least one non-zero value in its receptive field across all channels.
  // Positions not in this set will always produce zero output.
  vector<int> activePositions;
  int activeCount = outMatRow * outMatCol; // default: all positions active (dense)

  if (params.useInputSparseOpt) {
    computeActiveOutputPositions(inputMatrix, kernelH, kernelW, params.stride,
                                 outMatRow, outMatCol, activePositions);
    activeCount = activePositions.size();

    double activeFrac = (outMatRow * outMatCol > 0)
        ? 100.0 * activeCount / (outMatRow * outMatCol) : 0.0;
    cout << "  Active output positions:      "
         << activeCount << " / " << outMatRow * outMatCol
         << " (" << fixed << setprecision(1) << activeFrac << "% non-zero)"  << endl;
  }

  // Max channels per chunk for the SPARSE INPUT path (fewer cols per channel)
  int numOfMatPerRowSparse =
      (activeCount > 0 && (int64_t)floor((1.0 * numOfBits) / activeCount) < (int64_t)params.dim)
      ? (int)floor((1.0 * numOfBits) / activeCount)
      : (int)params.dim;
  if (numOfMatPerRowSparse < 1) numOfMatPerRowSparse = 1;

  // Choose which chunk size to use based on active optimization
  int effectiveMatPerRow = params.useInputSparseOpt ? numOfMatPerRowSparse : numOfMatPerRow;
  int effectiveHopSize   = params.useInputSparseOpt ? activeCount          : outMatRow * outMatCol;

  // -------------------------------------------------------------------------
  // Main convolution loop: one iteration per output filter
  // -------------------------------------------------------------------------
  chrono::duration<double, milli> hostElapsedTime =
      chrono::duration<double, milli>::zero();

  Image3D resultMatrix(outMatDim,
      vector<vector<int>>(outMatRow, vector<int>(outMatCol, 0)));

  for (uint64_t filterIdx = 0; filterIdx < (uint64_t)params.kernelDim; filterIdx++) {

    // outVector is the running accumulator across channel chunks.
    // For dense path:        size = outH*outW * inputDepth (all positions, all channels)
    // For input-sparse path: size = activeCount * inputDepth (only active positions)
    vector<int> outVector((uint64_t)effectiveHopSize * inputDepth, 0);
    vector<int> dstVec(effectiveHopSize, 0);

    for (uint64_t j = 0; j < (uint64_t)params.dim; j += effectiveMatPerRow) {
      uint64_t matChunk = min((uint64_t)(j + effectiveMatPerRow), (uint64_t)params.dim);
      // Build the merged decomposition matrix for channels [j, matChunk).
      // Each channel contributes numOfPIMRow rows and effectiveHopSize cols.
      // Channels are interleaved horizontally: merged[filterPos] = [ch_j | ch_{j+1} | ...].
      vector<vector<int>> mergedMat(numOfPIMRow);

      for (uint64_t ch = j; ch < matChunk; ch++) {
        if (params.useInputSparseOpt) {
          // Input sparsity path: only include active output positions
          vector<vector<int>> compDecomp;
          getCompressedDecompMatrix(params.row, params.column,
                                    kernelH, kernelW, params.stride,
                                    inputMatrix[ch], activePositions, compDecomp);
          for (int row = 0; row < numOfPIMRow; row++) {
            mergedMat[row].insert(mergedMat[row].end(),
                                  make_move_iterator(compDecomp[row].begin()),
                                  make_move_iterator(compDecomp[row].end()));
          }
        } else {
          // Dense path: all output positions
          vector<vector<int>> decompMat;
          getDecomposedMatrix(params.row, params.column,
                              kernelH, kernelW, params.stride,
                              inputMatrix[ch], decompMat);
          for (int row = 0; row < numOfPIMRow; row++) {
            mergedMat[row].insert(mergedMat[row].end(),
                                  make_move_iterator(decompMat[row].begin()),
                                  make_move_iterator(decompMat[row].end()));
          }
        }
      }

      if (params.moreDebugPrints) {
        cout << "Filter " << filterIdx << ", chunk ch=[" << j << "," << matChunk << "): "
             << "mergedMat = " << numOfPIMRow << " rows x "
             << (mergedMat.empty() ? 0 : mergedMat[0].size()) << " cols" << endl;
      }

      int tempcol = mergedMat.empty() ? 0 : (int)mergedMat[0].size();

      // Execute convolution on PIM.
      // Weight sparse opt: skip pimScaledAdd for zero-weight filter positions.
      performConv(kernelMatrix[filterIdx], mergedMat, outVector,
                  numOfPIMRow, tempcol,
                  params.useWeightSparseOpt,
                  params.moreDebugPrints);
    }

    // Reduce outVector across channel chunks via binary tree PIM adds.
    aggregate(outVector, dstVec, effectiveHopSize);

    // Store results into the output tensor.
    if (params.useInputSparseOpt) {
      // Input sparsity path: scatter compressed results to full output positions.
      // Positions not in activePositions were never touched and remain zero.
      for (int k = 0; k < activeCount; k++) {
        int flatIdx = activePositions[k];
        int r = flatIdx / outMatCol;
        int c = flatIdx % outMatCol;
        resultMatrix[filterIdx][r][c] = dstVec[k];
      }
    } else {
      // Dense path: reshape dstVec linearly into resultMatrix[filterIdx].
      int ddx = 0;
      for (int r = 0; r < outMatRow; r++) {
        for (int c = 0; c < outMatCol; c++) {
          resultMatrix[filterIdx][r][c] = dstVec[ddx++];
        }
      }
    }

    if (params.moreDebugPrints) {
      cout << "Filter " << filterIdx << " result:" << endl;
      printMatrix(resultMatrix[filterIdx]);
    }
  }

  // -------------------------------------------------------------------------
  // Verification
  // -------------------------------------------------------------------------
  if (params.shouldVerify) {
    VerifyWithCPU(inputMatrix, kernelMatrix,
                  params.padding, params.stride,
                  params.moreDebugPrints, resultMatrix);
  }

  // -------------------------------------------------------------------------
  // Stats
  // -------------------------------------------------------------------------
  pimShowStats();
  cout << "Host elapsed time: " << fixed << setprecision(3)
       << hostElapsedTime.count() << " ms." << endl;

  return 0;
}
