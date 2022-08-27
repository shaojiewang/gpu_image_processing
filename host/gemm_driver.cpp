#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include "image_generator.hpp"
#include "common.hpp"
#include "gemm_descrpitor.hpp"

using ADataType = float;
using BDataType = float;
using CDataType = float;
using AccDataType = float;

int main(int argc, char* argv[]){
    std::cout << "welcome to practice" << std::endl;

    // retrieve args
    if(argc > 4){
        return 0;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    std::cout << "M, N, K = {" << M << ", " << N << ", " << K << "}" << std::endl;

    // allocate tensor
    ADataType* APtr = static_cast<ADataType*>(malloc(M * K));
    BDataType* BPtr = static_cast<BDataType*>(malloc(N * K));
    CDataType* CPtr_ref = static_cast<CDataType*>(malloc(M * N));
    CDataType* CPtr_opt = static_cast<CDataType*>(malloc(M * N));

    gemm_descrpitor_cpu<ADataType, BDataType, CDataType, AccDataType> GemmDesc{M, N, K, APtr, BPtr, CPtr_ref, CPtr_opt};

    std::cout << "M, N, K = {" << GemmDesc.M << ", " << GemmDesc.N << ", " << GemmDesc.K << "}" << std::endl;


    return 0;
}