#pragma once

template<typename ADateType,
         typename BDateType,
         typename CDateType,
         typename AccDateType>
void gemm_reference(const ADateType* APtr, const BDateType* BPtr, CDateType* CPtr, int M, int N, int K)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            AccDateType sum = 0;
            for(int k = 0; k < K; k++)
            {
                sum += APtr[i * K + k] * BPtr[k * N + j];
            }
            CPtr[i * M + j] = sum;
        }
    }
}