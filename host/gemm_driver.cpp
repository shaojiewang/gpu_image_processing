#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include "image_generator.hpp"
#include "common.hpp"
#include "gemm_descrpitor.hpp"
#include "gemm_reference_code.hpp"
#include "host_memory.hpp"
#include "gemm_cpu_opt.hpp"

using ADataType = float;
using BDataType = float;
using CDataType = float;
using AccDataType = float;

int main(int argc, char* argv[]){
    std::cout << "welcome to practice" << std::endl;

    constexpr auto MemAlignment = 1024;

    // retrieve args
    if(argc != 5){
        return 0;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int times = atoi(argv[4]);

    std::cout << "M, N, K = {" << M << ", " << N << ", " << K << "}" << std::endl;

    // allocate tensor
    ADataType* APtr = static_cast<ADataType*>(aligned_mem_cpu(M * K * sizeof(ADataType), MemAlignment));
    BDataType* BPtr = static_cast<BDataType*>(aligned_mem_cpu(N * K * sizeof(BDataType), MemAlignment));
    CDataType* CPtr_ref = static_cast<CDataType*>(aligned_mem_cpu(M * N * sizeof(CDataType), MemAlignment));
    CDataType* CPtr_opt = static_cast<CDataType*>(aligned_mem_cpu(M * N * sizeof(CDataType), MemAlignment));

    using RandomGen = random_tensor_generator_2<ADataType>;

    random_gen<RandomGen, float>(APtr, M * K, RandomGen{-1, 1});
    random_gen<RandomGen, float>(BPtr, N * K, RandomGen{-1, 1});

    gemm_descrpitor_cpu<ADataType, BDataType, CDataType, AccDataType> GemmDesc{M, N, K, APtr, BPtr, CPtr_ref, CPtr_opt};

    std::cout << "M, N, K = {" << GemmDesc.M << ", " << GemmDesc.N << ", " << GemmDesc.K << "}" << std::endl;
    std::cout << "times = " << times << std::endl;

    gemm_reference<ADataType, BDataType, CDataType, AccDataType>(GemmDesc.APtr, 
                                                                 GemmDesc.BPtr,
                                                                 GemmDesc.CPtrRef,
                                                                 GemmDesc.M,
                                                                 GemmDesc.N,
                                                                 GemmDesc.K);

    auto mStart = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < times; i++)
    {
        gemm_reference<ADataType, BDataType, CDataType, AccDataType>(GemmDesc.APtr, 
                                                                     GemmDesc.BPtr,
                                                                     GemmDesc.CPtrRef,
                                                                     GemmDesc.M,
                                                                     GemmDesc.N,
                                                                     GemmDesc.K);
    }
    auto mStop = std::chrono::high_resolution_clock::now();

    float ms = static_cast<float>(
                   std::chrono::duration_cast<std::chrono::microseconds>(mStop - mStart).count()) *
               1e-3;

    std::cout << "ref gemm time: " << static_cast<float>(ms / times) << "ms." << std::endl;

    // opt code
    gemm_cpu_opt_reorder_loop<ADataType, BDataType, CDataType, AccDataType>(GemmDesc.APtr, 
                                                                            GemmDesc.BPtr,
                                                                            GemmDesc.CPtrOpt,
                                                                            GemmDesc.M,
                                                                            GemmDesc.N,
                                                                            GemmDesc.K);

    auto mStartOpt0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < times; i++)
    {
        gemm_cpu_opt_reorder_loop<ADataType, BDataType, CDataType, AccDataType>(GemmDesc.APtr, 
                                                                                GemmDesc.BPtr,
                                                                                GemmDesc.CPtrOpt,
                                                                                GemmDesc.M,
                                                                                GemmDesc.N,
                                                                                GemmDesc.K);
    }
    auto mStopOpt0 = std::chrono::high_resolution_clock::now();

    float msOpt0 = static_cast<float>(
                   std::chrono::duration_cast<std::chrono::microseconds>(mStopOpt0 - mStartOpt0).count()) *
               1e-3;

    std::cout << "cpu opt0 gemm time: " << static_cast<float>(msOpt0 / times) << "ms." << std::endl;

    // check err
    CDataType tol = static_cast<CDataType>(0.001);
    bool res = GemmDesc.checkErr(tol);

    if(res)
    {
        std::cout << "right res" << std::endl;
    }
    else
    {
        std::cout << "wrong res" << std::endl;
    }

    return 0;
}