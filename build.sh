#!/bin/sh
# host compilation
export TEST_KERNEL=image_processing
rm out/image_driver.exe
#rm out/${TEST_KERNEL}.hsaco

# build host
/opt/rocm/hip/bin/hipcc -std=c++14 host/image_driver.cpp -I host/ -o out/image_driver.exe

# build device
/opt/rocm/hip/bin/hipcc -x hip --cuda-gpu-arch=gfx908 \
--cuda-device-only -c -O3 \
device/gaussian_filter.cpp -o out/gaussian_filter.hsaco