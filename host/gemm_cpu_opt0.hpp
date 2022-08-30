#pragma once

template<typename ADataType,
         typename BDataType,
         typename CDataType,
         typename AccDataType>
void gemm_cpu_opt_reorder_loop(const ADataType* APtr, const BDataType* BPtr, CDataType* CPtr, int M, int N, int K)
{
    memset(CPtr, 0, M * N * sizeof(CDataType));
    for(int i = 0; i < M; i++)
    {
        for(int k = 0; k < K; k++)
        {
            ADataType AData = APtr[i * K + k];
            for(int j = 0; j < N; j++)
            {
                CPtr[i * N + j] += AData * BPtr[k * N + j];
            }
        }
    }
}
