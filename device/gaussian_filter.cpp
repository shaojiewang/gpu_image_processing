#include <hip/hip_runtime.h>
#include <stdio.h>

extern "C"
__global__ __launch_bounds__(256, 2)
void gaussian_filter(void* input, 
                     void* filter, 
                     void* output, 
                     int height, 
                     int width, 
                     int f_h, 
                     int f_w){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int j = bidx * blockDim.x + tidx;
    int i = bidy * blockDim.y + tidy;

    int f_h_2 = f_h / 2;
    int f_w_2 = f_w / 2;

    int k, l, cur_h, cur_w;
    float tmp = 0.f;

    float* tmp_input = (float*)input;
    float* tmp_filter = (float*)filter;
    float* tmp_output = (float*)output;
    
    for(k = 0; k < f_h; k++){
        int valid_h = 1;
        cur_h = i + k - f_h_2;
        if(cur_h < 0 || cur_h >= height){
            valid_h &= 0;
        }
        for(l = 0; l < f_w; l++){
            int valid_w = 1;
            cur_w = j + l - f_w_2;
            if(cur_w < 0 || cur_w >= width){
                valid_w &= 0;
            }
            if(valid_h && valid_w){
                tmp += tmp_filter[k * f_w + l] * tmp_input[cur_h * height + cur_w];
            }
        }
    }

    tmp_output[i * width + j] = tmp;
    
}