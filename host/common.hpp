#ifndef __COMMON_H
#define __COMMON_H

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <float.h>

#define HIP_CALL(call)                                                         \
    do{                                                                           \
        hipError_t err = call;                                                    \
        if(err != hipSuccess){                                                    \
            printf("line %d, [hiperror](%d), fail to call %s,(%s) \n", __LINE__,  \
            (int)err, #call, hipGetErrorString(err));                             \
            exit(1);                                                              \
        }                                                                         \
    }while(0)

#endif