#pragma once 

#include <immintrin.h>

template<typename ADataType,
         typename BDataType,
         typename CDataType,
         typename AccDataType,
         int MTile,
         int NTile,
         int KTile>
static inline void gemm_cpu_macro_tile_4x8(const ADataType* APtr, const BDataType* BPtr, CDataType* CPtr, int M, int N, int K)
{
    for(int i = 0; i < MTile; i += 1)
    {
        for(int k = 0; k < KTile; k++)
        {
            ADataType AData = APtr[i * K + k];
            for(int j = 0; j < NTile; j += 1)
            {   
                CPtr[i * N + j] += AData * BPtr[k * N + j];
            }
        }
    }
}

template<typename ADataType,
         typename BDataType,
         typename CDataType,
         typename AccDataType>
void gemm_cpu_opt_tile_l1_cache_avx512(const ADataType* APtr, const BDataType* BPtr, CDataType* CPtr, int M, int N, int K)
{
    // l1 data cache is 32KB
    constexpr int mc = 64;
    constexpr int nc = 64;
    constexpr int kc = 32;

    const ADataType* ATmp = APtr;
    const BDataType* BTmp = BPtr;
    CDataType* CTmp = CPtr;

    memset(CPtr, 0, M * N * sizeof(CDataType));
    for(int i = 0; i < M; i += mc)
    {
        for(int j = 0; j < N; j += nc)
        {
            for(int k = 0; k < K; k += kc)
            {
                gemm_cpu_macro_tile<ADataType, 
                                    BDataType, 
                                    CDataType, 
                                    AccDataType,
                                    mc,
                                    nc, 
                                    kc>
                    (ATmp, BTmp, CTmp, M, N, K);
                ATmp += kc;
                BTmp += kc * N;
            }
            CTmp += nc;
            BTmp += nc - (K * N);
            ATmp += -K;
        }
        ATmp += mc * K;
        CTmp += mc * N - N;
        BTmp = BPtr;
    }
}