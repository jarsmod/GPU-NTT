#!/bin/bash
#./bin/build-and-test-mtyp.sh 0
[[ "$1" ]] || { echo "Usage: $0 <0|1|2>"; exit 1; }
MTYP=$1
cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=$MTYP -B./bbuild/$MTYP && ( cd bbuild/$MTYP && make; ) && ./bbuild/$MTYP/test_ntt_gpu
