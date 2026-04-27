#!/usr/bin/env python3
"""
Sparse Convolution PIMeval Sweep
=================================
Sweeps M×M×1 input matrix sizes from ~1/10× to ~10× the PIM column capacity,
running each case RUNS times and recording per-run CPU overhead, PIM time, and
total time.  Results are written to sparse_conv_sweep_results.csv.

PIM Config  : configs/hbm/PIMeval_Bank_Rank1.cfg
  numBanks=128, numSubarrays=32, numCols=8192
  Column capacity = 128×32×8192 = 33,554,432 INT32 element slots
  1× PIM point → N² = 33,554,432 → N ≈ 5,792

Fixed parameters
  Kernel    : 3×3, no sparsity (-k 0.0)
  Input     : M×M×1, 98% sparsity (-a 0.98)
  Output filters : 1 (-z 1)
  Stride/padding : 1/1

PIM time reported = data-copy time + PIM-command execution time (from pimShowStats).
Total time = CPU overhead + PIM time.

RAM warning: the 10× point (N≈18 318) requires ~12 GB peak RAM.  Sizes estimated
to exceed MEM_LIMIT_GB are skipped automatically.
"""

import subprocess
import re
import csv
import math
import os
import sys

# ── Run configuration ────────────────────────────────────────────────────────
BINARY  = "./PIMbench/sparse-convolution/PIM/sparse_conv.out"
CONFIG  = "configs/hbm/PIMeval_Bank_Rank1.cfg"
OUT_CSV = "sparse_conv_sweep_results.csv"
RUNS    = 10
MEM_LIMIT_GB = 20          # skip sizes whose estimated peak RAM exceeds this

KERNEL_H        = 3
KERNEL_W        = 3
DEPTH           = 1
NUM_FILTERS     = 1        # -z: output filters (kept at 1 for 2-D input sweep)
PADDING         = 1
STRIDE          = 1
INPUT_SPARSITY  = 0.98
KERNEL_SPARSITY = 0.0

# ── PIM memory definition ────────────────────────────────────────────────────
# Column-addressable INT32 capacity of the Rank1 HBM config.
# Each column position holds one INT32 element; the full device has:
#   numBanks × numSubarrays × numCols = 128 × 32 × 8192 = 33,554,432 slots.
# Sweep fractions are relative to this value.
PIM_COL_CAPACITY = 128 * 32 * 8192   # = 33,554,432

FRACTIONS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
# ─────────────────────────────────────────────────────────────────────────────


def sweep_sizes():
    """Return list of (fraction, N) pairs for the sweep."""
    sizes = []
    for f in FRACTIONS:
        n = math.isqrt(int(PIM_COL_CAPACITY * f))
        sizes.append((f, n))
    return sizes


def run_once(N):
    cmd = [
        BINARY,
        "-r", str(N), "-c", str(N),
        "-d", str(DEPTH),
        "-z", str(NUM_FILTERS),
        "-l", str(KERNEL_H), "-w", str(KERNEL_W),
        "-k", str(KERNEL_SPARSITY),
        "-a", str(INPUT_SPARSITY),
        "-p", str(PADDING),
        "-s", str(STRIDE),
        "-o", CONFIG,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return result.stdout + result.stderr


def parse_output(output):
    """
    Parse pimShowStats + CPU breakdown output.

    Returns:
        (cpu_ms, pim_ms) where pim_ms = data_copy_ms + pim_cmd_ms,
        or (None, None) on parse failure.
    """
    cpu_overhead_ms = None
    data_copy_ms    = None
    pim_cmd_ms      = None

    in_data_copy = False
    in_pim_cmd   = False

    for line in output.splitlines():
        s = line.strip()

        if "Data Copy Stats:" in s:
            in_data_copy, in_pim_cmd = True, False
            continue
        if "PIM Command Stats:" in s:
            in_pim_cmd, in_data_copy = True, False
            continue

        # Data Copy TOTAL line:
        # TOTAL --------- : 45056 bytes       0.000147 ms Estimated Runtime
        if in_data_copy and s.startswith("TOTAL"):
            m = re.search(r"TOTAL\s+-+\s*:\s+\d+\s+bytes\s+([\d.eE+\-]+)\s+ms", s)
            if m:
                data_copy_ms = float(m.group(1))
            in_data_copy = False
            continue

        # PIM Command TOTAL line:
        # TOTAL --------- :          9       0.001413       ...
        if in_pim_cmd and s.startswith("TOTAL") and "PIM-CMD" not in s:
            m = re.search(r"TOTAL\s+-+\s*:\s+\d+\s+([\d.eE+\-]+)", s)
            if m:
                pim_cmd_ms = float(m.group(1))
            in_pim_cmd = False
            continue

        # CPU overhead summary line
        if "Total CPU overhead:" in s:
            m = re.search(r"Total CPU overhead:\s+([\d.eE+\-]+)\s+ms", s)
            if m:
                cpu_overhead_ms = float(m.group(1))

    if None in (cpu_overhead_ms, data_copy_ms, pim_cmd_ms):
        return None, None

    pim_ms = data_copy_ms + pim_cmd_ms
    return cpu_overhead_ms, pim_ms


def estimate_peak_ram_gb(N):
    """Rough peak RAM estimate (GB) for one benchmark run with depth=1, z=1."""
    # mergedMat = KH*KW rows × N² cols  (9 × N² int32)
    # + filterObject, matrixObject, input, output  (~4 × N² int32)
    return (9 + 4) * N * N * 4 / (1024 ** 3)


def fraction_label(f):
    if f >= 1.0:
        return f"{f:.0f}x"
    denom = int(round(1.0 / f))
    return f"1/{denom}x"


def main():
    if not os.path.exists(BINARY):
        print(f"ERROR: binary not found: {BINARY}", file=sys.stderr)
        sys.exit(1)

    sizes = sweep_sizes()
    n1x   = math.isqrt(PIM_COL_CAPACITY)

    print("=" * 72)
    print("Sparse Convolution PIMeval Sweep")
    print(f"  Config  : {CONFIG}")
    print(f"  Binary  : {BINARY}")
    print(f"  PIM column capacity : {PIM_COL_CAPACITY:,}  (N_1x ≈ {n1x})")
    print(f"  Sweep   : {len(sizes)} sizes × {RUNS} runs")
    print(f"  Kernel  : {KERNEL_H}×{KERNEL_W}, sparsity={KERNEL_SPARSITY:.0%}")
    print(f"  Input   : M×M×{DEPTH}, sparsity={INPUT_SPARSITY:.0%}, filters={NUM_FILTERS}")
    print(f"  Output  : {OUT_CSV}")
    print("=" * 72)

    csv_rows = []

    for frac, N in sizes:
        label    = fraction_label(frac)
        ram_est  = estimate_peak_ram_gb(N)
        mat_size = N * N

        print(f"\n{'─'*60}")
        print(f"  N={N}  ({N}×{N}={mat_size:,})  [{label} PIM]  est. RAM≈{ram_est:.1f} GB")

        if ram_est > MEM_LIMIT_GB:
            print(f"  SKIP: estimated RAM {ram_est:.1f} GB > {MEM_LIMIT_GB} GB limit.")
            continue

        cpu_samples = []
        pim_samples = []
        failed      = False

        for r in range(1, RUNS + 1):
            print(f"    run {r:2d}/{RUNS} ... ", end="", flush=True)
            try:
                out = run_once(N)
                cpu_ms, pim_ms = parse_output(out)

                if cpu_ms is None:
                    print("PARSE FAILED")
                    print("  --- raw output snippet ---")
                    print("\n".join(out.splitlines()[-30:]))
                    failed = True
                    break

                total_ms = cpu_ms + pim_ms
                print(f"CPU={cpu_ms:10.3f} ms  PIM={pim_ms:10.6f} ms  "
                      f"total={total_ms:10.3f} ms")
                cpu_samples.append(cpu_ms)
                pim_samples.append(pim_ms)

            except subprocess.TimeoutExpired:
                print("TIMEOUT (>900 s)")
                failed = True
                break
            except Exception as e:
                print(f"ERROR: {e}")
                failed = True
                break

        if failed or not cpu_samples:
            print(f"  Skipped N={N} due to errors.")
            continue

        avg_cpu   = sum(cpu_samples) / len(cpu_samples)
        avg_pim   = sum(pim_samples) / len(pim_samples)
        avg_total = avg_cpu + avg_pim

        print(f"  → AVERAGES  CPU={avg_cpu:.3f} ms  PIM={avg_pim:.6f} ms  "
              f"total={avg_total:.3f} ms  ({len(cpu_samples)} runs)")

        csv_rows.append({
            "matrix_width_length":    N,
            "matrix_size":            mat_size,
            "fraction_of_pim_memory": f"{frac:.2f}x",
            "avg_cpu_overhead_ms":    round(avg_cpu,   6),
            "avg_pim_time_ms":        round(avg_pim,   9),
            "avg_total_time_ms":      round(avg_total, 6),
        })

    if not csv_rows:
        print("\nNo successful runs — no CSV written.")
        sys.exit(1)

    fieldnames = [
        "matrix_width_length",
        "matrix_size",
        "fraction_of_pim_memory",
        "avg_cpu_overhead_ms",
        "avg_pim_time_ms",
        "avg_total_time_ms",
    ]
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'='*72}")
    print(f"Results written to: {OUT_CSV}")
    print(f"Columns: {', '.join(fieldnames)}")
    print()

    # Pretty-print summary table
    col_w = [max(len(fn), max(len(str(r[fn])) for r in csv_rows)) + 2
             for fn in fieldnames]
    header = "  ".join(fn.ljust(w) for fn, w in zip(fieldnames, col_w))
    print(header)
    print("-" * len(header))
    for row in csv_rows:
        print("  ".join(str(row[fn]).ljust(w) for fn, w in zip(fieldnames, col_w)))


if __name__ == "__main__":
    main()
