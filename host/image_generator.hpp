#pragma once

#include <stdio.h>
#include <stdlib.h>

template <typename T>
struct random_tensor_generator_2
{
    /* data */
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        return static_cast<T>(std::rand() % (max_value - min_value) + min_value);
    }
};

template<typename F, typename T>
void random_gen(void* images, size_t size, F f){
    T* tmp = (T*)images;
    for(size_t i = 0; i < size; i++){
        //tmp[i] = 1.f;
        tmp[i] = static_cast<T>(f());
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
