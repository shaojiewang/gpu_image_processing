#pragma once 

#include <immintrin.h>

template<typename ADataType,
         typename BDataType,
         typename CDataType,
         typename AccDataType>
static inline void gemm_cpu_micro_tile_4x32x32(const ADataType* APtr, const BDataType* BPtr, CDataType* CPtr, int M, int N, int K)
{
    __m512 v_b_n0_f_16, v_b_n1_f_16;
    __m512 v_c_m0n0_f_16, v_c_m1n0_f_16, v_c_m2n0_f_16, v_c_m3n0_f_16;
    __m512 v_c_m0n1_f_16, v_c_m1n1_f_16, v_c_m2n1_f_16, v_c_m3n1_f_16;
    __m512 v_a_m0_f_16, v_a_m1_f_16, v_a_m2_f_16, v_a_m3_f_16;
    __m128 v_a_m0_f_4, v_a_m1_f_4, v_a_m2_f_4, v_a_m3_f_4;

    CDataType* CTmp0 = CPtr;
    CDataType* CTmp1 = CTmp0 + N;
    CDataType* CTmp2 = CTmp1 + N;
    CDataType* CTmp3 = CTmp2 + N;

    CDataType* CTmp4 = CTmp0 + 16;
    CDataType* CTmp5 = CTmp1 + 16;
    CDataType* CTmp6 = CTmp2 + 16;
    CDataType* CTmp7 = CTmp3 + 16;

    const ADataType* ATmp = APtr;

    v_c_m0n0_f_16 = _mm512_load_ps(CTmp0);
    v_c_m1n0_f_16 = _mm512_load_ps(CTmp1);
    v_c_m2n0_f_16 = _mm512_load_ps(CTmp2);
    v_c_m3n0_f_16 = _mm512_load_ps(CTmp3);

    v_c_m0n1_f_16 = _mm512_load_ps(CTmp4);
    v_c_m1n1_f_16 = _mm512_load_ps(CTmp5);
    v_c_m2n1_f_16 = _mm512_load_ps(CTmp6);
    v_c_m3n1_f_16 = _mm512_load_ps(CTmp7); 

    for(int k = 0; k < 32; k++)
    {
        ATmp = APtr;
        v_a_m0_f_4 = _mm_broadcast_ss(ATmp); ATmp += K;
        v_a_m1_f_4 = _mm_broadcast_ss(ATmp); ATmp += K;
        v_a_m2_f_4 = _mm_broadcast_ss(ATmp); ATmp += K;
        v_a_m3_f_4 = _mm_broadcast_ss(ATmp);
        v_a_m0_f_16 = _mm512_broadcast_f32x2(v_a_m0_f_4);
        v_a_m1_f_16 = _mm512_broadcast_f32x2(v_a_m1_f_4);
        v_a_m2_f_16 = _mm512_broadcast_f32x2(v_a_m2_f_4);
        v_a_m3_f_16 = _mm512_broadcast_f32x2(v_a_m3_f_4);

        v_b_n0_f_16 = _mm512_load_ps(BPtr); BPtr += 16;
        v_b_n1_f_16 = _mm512_load_ps(BPtr); BPtr += N - 16;

        v_c_m0n0_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n0_f_16, v_c_m0n0_f_16);
        v_c_m1n0_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n0_f_16, v_c_m1n0_f_16);
        v_c_m2n0_f_16 = _mm512_fmadd_ps(v_a_m2_f_16, v_b_n0_f_16, v_c_m2n0_f_16);
        v_c_m3n0_f_16 = _mm512_fmadd_ps(v_a_m3_f_16, v_b_n0_f_16, v_c_m3n0_f_16);

        v_c_m0n1_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n1_f_16, v_c_m0n1_f_16);
        v_c_m1n1_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n1_f_16, v_c_m1n1_f_16);
        v_c_m2n1_f_16 = _mm512_fmadd_ps(v_a_m2_f_16, v_b_n1_f_16, v_c_m2n1_f_16);
        v_c_m3n1_f_16 = _mm512_fmadd_ps(v_a_m3_f_16, v_b_n1_f_16, v_c_m3n1_f_16);

        APtr++;
    }
    _mm512_store_ps(CTmp0, v_c_m0n0_f_16);
    _mm512_store_ps(CTmp1, v_c_m1n0_f_16);
    _mm512_store_ps(CTmp2, v_c_m2n0_f_16);
    _mm512_store_ps(CTmp3, v_c_m3n0_f_16);

    _mm512_store_ps(CTmp4, v_c_m0n1_f_16);
    _mm512_store_ps(CTmp5, v_c_m1n1_f_16);
    _mm512_store_ps(CTmp6, v_c_m2n1_f_16);
    _mm512_store_ps(CTmp7, v_c_m3n1_f_16);
}

template<typename ADataType,
         typename BDataType,
         typename CDataType,
         typename AccDataType,
         int MTile,
         int NTile,
         int KTile>
static inline void gemm_cpu_macro_tile_M_N_K(const ADataType* APtr, const BDataType* BPtr, CDataType* CPtr, int M, int N, int K)
{
    const ADataType* ATmp = APtr;
    const BDataType* BTmp = BPtr;
    CDataType* CTmp = CPtr;
    for(int i = 0; i < MTile; i += 4)
    {
        for(int j = 0; j < NTile; j += 32)
        {   
            gemm_cpu_micro_tile_4x32x32<ADataType, BDataType, CDataType, AccDataType>(ATmp, BTmp, CTmp, M, N, K);
            BTmp += 32;
            CTmp += 32;
        }
        ATmp += 4 * K;
        BTmp = BPtr;
        CTmp += 4 * N - NTile;
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
                gemm_cpu_macro_tile_M_N_K<ADataType, 
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