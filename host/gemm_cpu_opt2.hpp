#pragma once 

template<typename ADataType,
         typename BDataType,
         typename CDataType,
         typename AccDataType,
         int MTile,
         int NTile,
         int KTile>
static inline void gemm_cpu_macro_tile(const ADataType* APtr, const BDataType* BPtr, CDataType* CPtr, int M, int N, int K)
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
void gemm_cpu_opt_tile_l1_cache_avx(const ADataType* APtr, const BDataType* BPtr, CDataType* CPtr, int M, int N, int K)
{
    // l1 data cache is 32KB
    constexpr int mr = 64;
    constexpr int nr = 64;
    constexpr int kc = 32;

    const ADataType* ATmp = APtr;
    const BDataType* BTmp = BPtr;
    CDataType* CTmp = CPtr;

    memset(CPtr, 0, M * N * sizeof(CDataType));
    for(int i = 0; i < M; i += mr)
    {
        for(int j = 0; j < N; j += nr)
        {
            for(int k = 0; k < K; k += kc)
            {
                gemm_cpu_macro_tile<ADataType, 
                                    BDataType, 
                                    CDataType, 
                                    AccDataType,
                                    mr,
                                    nr, 
                                    kc>
                    (ATmp, BTmp, CTmp, M, N, K);
                ATmp += kc;
                BTmp += kc * N;
            }
            CTmp += nr;
            BTmp += nr - (K * N);
            ATmp += -K;
        }
        ATmp += mr * K;
        CTmp += mr * N - N;
        BTmp = BPtr;
    }
}