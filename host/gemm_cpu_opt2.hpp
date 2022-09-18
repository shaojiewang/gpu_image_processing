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
    __m512 v_b_f_16;
    __m512 v_c_i0_f_16, v_c_i1_f_16, v_c_i2_f_16, v_c_i3_f_16;
    __m512 v_a_i0_f_16, v_a_i1_f_16, v_a_i2_f_16, v_a_i3_f_16;
    __m128 v_a_f_i0_4, v_a_f_i1_4, v_a_f_i2_4, v_a_f_i3_4;
    for(int k = 0; k < KTile; k += 1)
    {
        for(int i = 0; i < MTile; i += 4)
        {
            //ADataType AData = APtr[i * K + k];
            v_a_f_i0_4 = _mm_broadcast_ss(APtr + i * K + k);
            v_a_f_i1_4 = _mm_broadcast_ss(APtr + (i + 1) * K + k);
            v_a_f_i2_4 = _mm_broadcast_ss(APtr + (i + 2) * K + k);
            v_a_f_i3_4 = _mm_broadcast_ss(APtr + (i + 3) * K + k);
            v_a_i0_f_16 = _mm512_broadcast_f32x2(v_a_f_i0_4);
            v_a_i1_f_16 = _mm512_broadcast_f32x2(v_a_f_i1_4);
            v_a_i2_f_16 = _mm512_broadcast_f32x2(v_a_f_i2_4);
            v_a_i3_f_16 = _mm512_broadcast_f32x2(v_a_f_i3_4);
            for(int j = 0; j < NTile; j += 16)
            {   
                v_b_f_16 = _mm512_load_ps(BPtr + k * N + j);

                v_c_i0_f_16 = _mm512_load_ps(CPtr + i * N + j);
                v_c_i1_f_16 = _mm512_load_ps(CPtr + (i + 1) * N + j);
                v_c_i2_f_16 = _mm512_load_ps(CPtr + (i + 2) * N + j);
                v_c_i3_f_16 = _mm512_load_ps(CPtr + (i + 3) * N + j);

                v_c_i0_f_16 = _mm512_fmadd_ps(v_a_i0_f_16, v_b_f_16, v_c_i0_f_16);
                v_c_i1_f_16 = _mm512_fmadd_ps(v_a_i1_f_16, v_b_f_16, v_c_i1_f_16);
                v_c_i2_f_16 = _mm512_fmadd_ps(v_a_i2_f_16, v_b_f_16, v_c_i2_f_16);
                v_c_i3_f_16 = _mm512_fmadd_ps(v_a_i3_f_16, v_b_f_16, v_c_i3_f_16);

                _mm512_store_ps(CPtr + i * N + j, v_c_i0_f_16);
                _mm512_store_ps(CPtr + (i + 1) * N + j, v_c_i1_f_16);
                _mm512_store_ps(CPtr + (i + 2) * N + j, v_c_i2_f_16);
                _mm512_store_ps(CPtr + (i + 3) * N + j, v_c_i3_f_16);
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