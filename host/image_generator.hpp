#ifndef __IMAGE_GENERATOR__
#define __IMAGE_GENERATOR__

#include <stdio.h>
#include <stdlib.h>

void random_gen(void* images, size_t size){
    float* tmp = (float*)images;
    for(size_t i = 0; i < size; i++){
        tmp[i] = 1.f;
    }
}

void gen_gaussian_filter(void* filter, size_t f_h, size_t f_w){
    float* tmp = (float*)filter;
    for(size_t i = 0; i < f_h; i++){
        for(size_t j = 0; j < f_w; j++){
            tmp[i] = 1.f;
        }
    }
}
#endif