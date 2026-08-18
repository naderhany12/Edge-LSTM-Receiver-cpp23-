#!/bin/bash

# Script: optimization_sweep.sh
# Purpose: Compiler Optimization Sweep for RISC-V (rv64gcv) on QEMU

set -e

# Set up the environment
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'
YELLOW='\033[1;33m'

OPTIMIZATION_FLAGS=( "-O0" "-O2" "-O3" "-Os" "-Ofast")
TOOLCHAIN_FILE="cmake/riscv64-toolchain.cmake"

echo -e "${CYAN}   Starting RISC-V Compiler Optimization Sweep      ${NC}"
echo -e "${CYAN}   -----------------------------------------------${NC}"

if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo -e "${YELLOW}[ERROR] Toolchain file '$TOOLCHAIN_FILE' not found!${NC}"
    exit 1
fi

declare -A BINARY_SIZES
declare -A LATENCIES

for FLAGS in "${OPTIMIZATION_FLAGS[@]}"; do
    BUILD_DIR="build_sweep_${FLAGS//-/}"
    echo -e "${GREEN}---> Testing Optimization Flag: ${FLAG}${NC}"

    # Passing the flag to the toolchain file
    cmake -S . -B "$BUILD_DIR"\ -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \-DCMAKE_CXX_FLAGS="${FLAG} -fopt-info-vec-optimized" \ /dev/null

    # compile the code
    cmake --build "$BUILD_DIR" > /dev/null 2>&1

     BINARY_PATH="$BUILD_DIR/lstm_inference"

    SIZE_BYTES=$(riscv64-linux-gnu-size "$BINARY_PATH" | tail -n 1 | awk '{print $1}')
    BINARY_SIZES["$FLAG"]=$SIZE_BYTES

    EXEC_OUTPUT=$(qemu-riscv64 -cpu rv64,v=true,vlen=128 "$BINARY_PATH")
    LATENCY=$(echo "$EXEC_OUTPUT" | grep "Pure Inference Latency" | awk '{print $(NF-1)}')
    LATENCIES["$FLAG"]=$LATENCY

    echo -e "     Binary .text Size : ${SIZE_BYTES} bytes"
    echo -e "     Inference Latency : ${LATENCY} us\n"
done

# (The Final Matrix Report)

echo -e "${CYAN}====================================================${NC}"
echo -e "${CYAN}          OPTIMIZATION SWEEP RESULTS MATRIX         ${NC}"
echo -e "${CYAN}====================================================${NC}"
printf "| %-10s | %-18s | %-18s |\n" "Opt Level" "Code Size (.text)" "Pure Latency (us)" "Auto-Vectorized"
printf "|------------|--------------------|--------------------|\n"

for FLAG in "${OPTIMIZATION_FLAGS[@]}"; do
    printf "| %-10s | %-18s | %-18s |\n" "$FLAG" "${BINARY_SIZES[$FLAG]} Bytes" "${LATENCIES[$FLAG]} us"
done
echo -e "${CYAN}====================================================${NC}\n"

