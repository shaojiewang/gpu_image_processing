#pragma once 

#include <immintrin.h>

template<typename ADataType,
         typename BDataType,
         typename CDataType,
         typename AccDataType>
static inline void gemm_cpu_micro_tile_8x48x32(const ADataType* APtr, const BDataType* BPtr, CDataType* CPtr, int M, int N, int K)
{
    __m512 v_b_n0_f_16, v_b_n1_f_16, v_b_n2_f_16;
    __m512 v_c_m0n0_f_16, v_c_m1n0_f_16, v_c_m2n0_f_16, v_c_m3n0_f_16;
    __m512 v_c_m4n0_f_16, v_c_m5n0_f_16, v_c_m6n0_f_16, v_c_m7n0_f_16;
    __m512 v_c_m0n1_f_16, v_c_m1n1_f_16, v_c_m2n1_f_16, v_c_m3n1_f_16;
    __m512 v_c_m4n1_f_16, v_c_m5n1_f_16, v_c_m6n1_f_16, v_c_m7n1_f_16;
    __m512 v_c_m0n2_f_16, v_c_m1n2_f_16, v_c_m2n2_f_16, v_c_m3n2_f_16;
    __m512 v_c_m4n2_f_16, v_c_m5n2_f_16, v_c_m6n2_f_16, v_c_m7n2_f_16;
    __m512 v_a_m0_f_16, v_a_m1_f_16;

    const ADataType* ATmp = APtr;

    v_c_m0n0_f_16 = _mm512_load_ps(CPtr + 0 * N + 0 * 16);
    v_c_m0n1_f_16 = _mm512_load_ps(CPtr + 0 * N + 1 * 16);
    v_c_m0n2_f_16 = _mm512_load_ps(CPtr + 0 * N + 2 * 16);

    v_c_m1n0_f_16 = _mm512_load_ps(CPtr + 1 * N + 0 * 16);
    v_c_m1n1_f_16 = _mm512_load_ps(CPtr + 1 * N + 1 * 16);
    v_c_m1n2_f_16 = _mm512_load_ps(CPtr + 1 * N + 2 * 16);

    v_c_m2n0_f_16 = _mm512_load_ps(CPtr + 2 * N + 0 * 16);
    v_c_m2n1_f_16 = _mm512_load_ps(CPtr + 2 * N + 1 * 16);
    v_c_m2n2_f_16 = _mm512_load_ps(CPtr + 2 * N + 2 * 16);

    v_c_m3n0_f_16 = _mm512_load_ps(CPtr + 3 * N + 0 * 16);
    v_c_m3n1_f_16 = _mm512_load_ps(CPtr + 3 * N + 1 * 16);
    v_c_m3n2_f_16 = _mm512_load_ps(CPtr + 3 * N + 2 * 16); 

    v_c_m4n0_f_16 = _mm512_load_ps(CPtr + 4 * N + 0 * 16);
    v_c_m4n1_f_16 = _mm512_load_ps(CPtr + 4 * N + 1 * 16);
    v_c_m4n2_f_16 = _mm512_load_ps(CPtr + 4 * N + 2 * 16);

    v_c_m5n0_f_16 = _mm512_load_ps(CPtr + 5 * N + 0 * 16);
    v_c_m5n1_f_16 = _mm512_load_ps(CPtr + 5 * N + 1 * 16);
    v_c_m5n2_f_16 = _mm512_load_ps(CPtr + 5 * N + 2 * 16);
    v_c_m6n0_f_16 = _mm512_load_ps(CPtr + 6 * N + 0 * 16);
    v_c_m6n1_f_16 = _mm512_load_ps(CPtr + 6 * N + 1 * 16);
    v_c_m6n2_f_16 = _mm512_load_ps(CPtr + 6 * N + 2 * 16);

    v_c_m7n0_f_16 = _mm512_load_ps(CPtr + 7 * N + 0 * 16);
    v_c_m7n1_f_16 = _mm512_load_ps(CPtr + 7 * N + 1 * 16); 
    v_c_m7n2_f_16 = _mm512_load_ps(CPtr + 7 * N + 2 * 16); 

    for(int k = 0; k < 32; k++)
    {
        ATmp = APtr;
        v_a_m0_f_16 = _mm512_set1_ps(ATmp[0 * K]);

        v_b_n0_f_16 = _mm512_loadu_ps(BPtr); BPtr += 16;
        v_b_n1_f_16 = _mm512_loadu_ps(BPtr); BPtr += 16;
        v_b_n2_f_16 = _mm512_loadu_ps(BPtr); BPtr += N - 32;

        v_a_m1_f_16 = _mm512_set1_ps(ATmp[1 * K]);

        v_c_m0n0_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n0_f_16, v_c_m0n0_f_16);
        v_c_m0n1_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n1_f_16, v_c_m0n1_f_16);
        v_c_m0n2_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n2_f_16, v_c_m0n2_f_16);

        v_a_m0_f_16 = _mm512_set1_ps(ATmp[2 * K]);
        
        v_c_m1n0_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n0_f_16, v_c_m1n0_f_16);
        v_c_m1n1_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n1_f_16, v_c_m1n1_f_16);
        v_c_m1n2_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n2_f_16, v_c_m1n2_f_16);

        v_a_m1_f_16 = _mm512_set1_ps(ATmp[3 * K]);

        v_c_m2n0_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n0_f_16, v_c_m2n0_f_16);
        v_c_m2n1_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n1_f_16, v_c_m2n1_f_16);
        v_c_m2n2_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n2_f_16, v_c_m2n2_f_16);

        v_a_m0_f_16 = _mm512_set1_ps(ATmp[4 * K]);

        v_c_m3n0_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n0_f_16, v_c_m3n0_f_16);
        v_c_m3n1_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n1_f_16, v_c_m3n1_f_16);
        v_c_m3n2_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n2_f_16, v_c_m3n2_f_16);

        v_a_m1_f_16 = _mm512_set1_ps(ATmp[5 * K]);

        v_c_m4n0_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n0_f_16, v_c_m4n0_f_16);
        v_c_m4n1_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n1_f_16, v_c_m4n1_f_16);
        v_c_m4n2_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n2_f_16, v_c_m4n2_f_16);

        v_a_m0_f_16 = _mm512_set1_ps(ATmp[6 * K]);

        v_c_m5n0_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n0_f_16, v_c_m5n0_f_16);
        v_c_m5n1_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n1_f_16, v_c_m5n1_f_16);
        v_c_m5n2_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n2_f_16, v_c_m5n2_f_16);

        v_a_m1_f_16 = _mm512_set1_ps(ATmp[7 * K]);

        v_c_m6n0_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n0_f_16, v_c_m6n0_f_16);
        v_c_m6n1_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n1_f_16, v_c_m6n1_f_16);
        v_c_m6n2_f_16 = _mm512_fmadd_ps(v_a_m0_f_16, v_b_n2_f_16, v_c_m6n2_f_16);

        v_c_m7n0_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n0_f_16, v_c_m7n0_f_16);
        v_c_m7n1_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n1_f_16, v_c_m7n1_f_16);
        v_c_m7n2_f_16 = _mm512_fmadd_ps(v_a_m1_f_16, v_b_n2_f_16, v_c_m7n2_f_16);

        APtr++;
    }
    
    _mm512_storeu_ps(CPtr + 0 * N + 0 * 16, v_c_m0n0_f_16);
    _mm512_storeu_ps(CPtr + 0 * N + 1 * 16, v_c_m0n1_f_16);
    _mm512_storeu_ps(CPtr + 0 * N + 2 * 16, v_c_m0n2_f_16);

    _mm512_storeu_ps(CPtr + 1 * N + 0 * 16, v_c_m1n0_f_16);
    _mm512_storeu_ps(CPtr + 1 * N + 1 * 16, v_c_m1n1_f_16);
    _mm512_storeu_ps(CPtr + 1 * N + 2 * 16, v_c_m1n2_f_16);

    _mm512_storeu_ps(CPtr + 2 * N + 0 * 16, v_c_m2n0_f_16);
    _mm512_storeu_ps(CPtr + 2 * N + 1 * 16, v_c_m2n1_f_16);
    _mm512_storeu_ps(CPtr + 2 * N + 2 * 16, v_c_m2n2_f_16);

    _mm512_storeu_ps(CPtr + 3 * N + 0 * 16, v_c_m3n0_f_16);
    _mm512_storeu_ps(CPtr + 3 * N + 1 * 16, v_c_m3n1_f_16);
    _mm512_storeu_ps(CPtr + 3 * N + 2 * 16, v_c_m3n2_f_16);

    _mm512_storeu_ps(CPtr + 4 * N + 0 * 16, v_c_m4n0_f_16);
    _mm512_storeu_ps(CPtr + 4 * N + 1 * 16, v_c_m4n1_f_16);
    _mm512_storeu_ps(CPtr + 4 * N + 2 * 16, v_c_m4n2_f_16);

    _mm512_storeu_ps(CPtr + 5 * N + 0 * 16, v_c_m5n0_f_16);
    _mm512_storeu_ps(CPtr + 5 * N + 1 * 16, v_c_m5n1_f_16);
    _mm512_storeu_ps(CPtr + 5 * N + 2 * 16, v_c_m5n2_f_16);

    _mm512_storeu_ps(CPtr + 6 * N + 0 * 16, v_c_m6n0_f_16);
    _mm512_storeu_ps(CPtr + 6 * N + 1 * 16, v_c_m6n1_f_16);
    _mm512_storeu_ps(CPtr + 6 * N + 2 * 16, v_c_m6n2_f_16);

    _mm512_storeu_ps(CPtr + 7 * N + 0 * 16, v_c_m7n0_f_16);
    _mm512_storeu_ps(CPtr + 7 * N + 1 * 16, v_c_m7n1_f_16);
    _mm512_storeu_ps(CPtr + 7 * N + 2 * 16, v_c_m7n2_f_16);
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

    constexpr auto mr = 8;
    constexpr auto nr = 48;

    for(int i = 0; i < MTile; i += mr)
    {
        for(int j = 0; j < NTile; j += nr)
        {   
            gemm_cpu_micro_tile_8x48x32<ADataType, BDataType, CDataType, AccDataType>(ATmp, BTmp, CTmp, M, N, K);
            BTmp += nr;
            CTmp += nr;
        }
        ATmp += mr * K;
        BTmp = BPtr;
        CTmp += mr * N - NTile;
    }
}

template<typename ADataType,
         typename BDataType,
         typename CDataType,
         typename AccDataType>
void gemm_cpu_opt_tile_l1_cache_avx512(const ADataType* APtr, const BDataType* BPtr, CDataType* CPtr, int M, int N, int K)
{
    // l1 data cache is 32KB
    constexpr int mc = 16;
    constexpr int nc = 144;
    constexpr int kc = 32;

    const ADataType* ATmp = APtr;
    const BDataType* BTmp = BPtr;
    CDataType* CTmp = CPtr;

    memset(CPtr, 0, M * N * sizeof(CDataType));
    for(int i = 0; i < M; i += mc)
    {
        for(int k = 0; k < K; k += kc)
        {
            for(int j = 0; j < N; j += nc)
            {
                gemm_cpu_macro_tile_M_N_K<ADataType, 
                                         BDataType, 
                                         CDataType, 
                                         AccDataType,
                                         mc,
                                         nc, 
                                         kc>
                    (ATmp, BTmp, CTmp, M, N, K);
                
                BTmp += nc;
                CTmp += nc;
            }
            ATmp += kc;
            BTmp += kc * N - N;
            CTmp -= N;
        }
        ATmp += mc * K - K;
        CTmp += mc * N;
        BTmp = BPtr;
    }
}