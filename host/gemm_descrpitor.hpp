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

    bool checkErr(CDataType tol)
    {
      int err_num = 0;
      bool res = true;
      CDataType max_err = -std::numeric_limits<CDataType>::infinity();
      for(int i = 0; i < M; i++)
      {
        for(int j = 0; j < N; j++)
        {
          int index = i * N + j;
          CDataType cur_err = std::abs(CPtrOpt[index] - CPtrRef[index]);
          if(cur_err > tol)
          {
            err_num++;
            if(cur_err > max_err)
            {
              max_err = cur_err;
            }
            res = false;
          }
          if(err_num < 10)
          {
            if(!res)
            {
              std::cout << "At [" << i << ", " << j << "]: " << "ref = " 
                        << CPtrRef[index] << ", opt = " << CPtrOpt[index] 
                        << std::endl;
            }
          }
        }
      }
      return res;
    }
};
