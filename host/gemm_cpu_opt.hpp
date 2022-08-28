#pragma once

template<typename ADateType,
         typename BDateType,
         typename CDateType,
         typename AccDateType>
void gemm_cpu_opt_reorder_loop(const ADateType* APtr, const BDateType* BPtr, CDateType* CPtr, int M, int N, int K)
{
    memset(CPtr, 0, M * N * sizeof(CDateType));
    for(int i = 0; i < M; i++)
    {
        for(int k = 0; k < K; k++)
        {
            ADateType AData = APtr[i * K + k];
            for(int j = 0; j < N; j++)
            {
                CPtr[i * M + j] += AData * BPtr[k * N + j];
            }
        }
    }
}