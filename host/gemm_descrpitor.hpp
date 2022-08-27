#pragma once

template<typename ADataType,
         typename BDataType,
         typename CDataType,
         typename AccDataType>
struct gemm_descrpitor_cpu
{
    /* data */
    int M;
    int N;
    int K;

    ADataType* APtr;
    BDataType* BPtr;
    CDataType* CPtrRef;
    CDataType* CPtrOpt;

    gemm_descrpitor_cpu(int M_,
                        int N_,
                        int K_,
                        ADataType* APtr_,
                        BDataType* BPtr_,
                        CDataType* CPtrRef_,
                        CDataType* CPtrOpt_)
    : M(M_),
      N(N_),
      K(K_),
      APtr(APtr_),
      BPtr(BPtr_),
      CPtrRef(CPtrRef_),
      CPtrOpt(CPtrOpt_)
    {

    }
};
