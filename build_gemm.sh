#!/bin/sh
# host compilation
export TEST_KERNEL=gemm
rm out/gemm_driver.exe
#rm out/${TEST_KERNEL}.hsaco

# build host
/opt/rocm/hip/bin/hipcc -std=c++14 host/gemm_driver.cpp -I host/ -o out/gemm_driver.exe