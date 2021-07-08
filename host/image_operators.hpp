#ifndef __IMAGE_OPRATOR__
#define __IMAGE_OPRATOR__

typedef struct 
{
    /* data */
    void* input;
    void* filter;
    void* output;
    int height;
    int width;
    int f_h;
    int f_w;
}__attribute__((packed)) gaussian_kernel_args;


#endif
