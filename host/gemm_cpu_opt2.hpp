#pragma once 

#include <immintrin.h>

template<typename ADataType,
         typename BDataType,
         typename CDataType,
         typename AccDataType,
         int MTile,
         int NTile,
         int KTile>
static inline void gemm_cpu_macro_tile_4x16(const ADataType* APtr, const BDataType* BPtr, CDataType* CPtr, int M, int N, int K)
{
    //const BDataType* BTmp = BPtr;
    __m512 v_b_k0_f_16, v_b_k1_f_16, v_b_k2_f_16, v_b_k3_f_16, v_c_f_16;
    __m128* p_v_a_f_4;
    for(int i = 0; i < MTile; i += 1)
    {
        for(int k = 0; k < KTile; k += 4)
        {
            //ADataType AData = APtr[i * K + k];
            p_v_a_f_4 = static_cast<__m128*>(APtr + i * K + k);
            for(int j = 0; j < NTile; j += 16)
            {   
                v_b_k0_f_16 = _mm512_load_ps(BPtr + k * N + j);
                v_b_k1_f_16 = _mm512_load_ps(BPtr + (k + 1) * N + j);
                v_b_k2_f_16 = _mm512_load_ps(BPtr + (k + 2) * N + j);
                v_b_k3_f_16 = _mm512_load_ps(BPtr + (k + 3 ) * N + j);
                v_c_f_16 = _mm512_load_ps(CPtr + i * N + j);
                v_c_f_16 = _mm512_4fmadd_ps(v_c_f_16, v_b_k0_f_16, v_b_k1_f_16, v_b_k2_f_16, v_b_k3_f_16, p_v_a_f_4);
                _mm512_store_ps(CPtr + i * N + j, v_c_f_16);
                //CPtr[i * N + j] += AData * BPtr[k * N + j];
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
                gemm_cpu_macro_tile_4x16<ADataType, 
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