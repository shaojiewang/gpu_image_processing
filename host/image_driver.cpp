#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include "image_generator.hpp"
#include "common.hpp"
#include "image_operators.hpp"

int main(int argc, char* argv[]){
    std::cout << "welcome to practice" << std::endl;

    // allocation of tensor
    void* image_data = nullptr;
    void* filter = nullptr;
    void* output_data = nullptr;

    void* image_data_device = nullptr;
    void* filter_device = nullptr;
    void* output_data_device = nullptr;

    void* check_output = nullptr;

    // retrieve args
    if(argc > 3){
        return 0;
    }
    int height = atoi(argv[1]);
    int width  = atoi(argv[2]);

    //std::cout << "height=" << height << ", width=" << width << std::endl;

    // ptr allocation
    image_data = malloc(height * width * sizeof(float));
    output_data = malloc(height * width * sizeof(float));
    check_output = malloc(height * width * sizeof(float));
    filter = malloc(3 * 3 * sizeof(int));

    HIP_CALL(hipMalloc(&image_data_device, height * width * sizeof(float))); 
    HIP_CALL(hipMalloc(&output_data_device, height * width * sizeof(float))); 
    HIP_CALL(hipMalloc(&filter_device, 3 * 3 * sizeof(int))); 

    // random generate input and filter
    random_gen(image_data, (size_t)height * width);
    gen_gaussian_filter(filter, 3, 3);

    HIP_CALL(hipMemcpy(image_data_device, image_data, height * width * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(filter_device, filter, 3 * 3 * sizeof(float), hipMemcpyHostToDevice));

    // call a device kernel
    hipModule_t gaussian_filer_module;
    hipFunction_t gaussian_filer_func;
    std::string gaussian_filer_hsaco = "gaussian_filter.hsaco";
    std::string gaussian_filer_func_name = "gaussian_filter";
    HIP_CALL(hipModuleLoad(&gaussian_filer_module, gaussian_filer_hsaco.c_str()));
    HIP_CALL(hipModuleGetFunction(&gaussian_filer_func, gaussian_filer_module, gaussian_filer_func_name.c_str()));

    gaussian_kernel_args g_kargs;
    size_t arg_size = sizeof(gaussian_kernel_args);
    g_kargs.input = image_data_device;
    g_kargs.filter = filter_device;
    g_kargs.output = output_data_device;
    g_kargs.height = height;
    g_kargs.width = width;
    g_kargs.f_h = 3;
    g_kargs.f_w = 3;


    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &g_kargs,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                        HIP_LAUNCH_PARAM_END};
    float ms = .0;

    hipEvent_t start;
    hipEvent_t stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    std::vector<size_t> grid_size(3, 1);
    std::vector<size_t> block_size(3, 1);

    // for hipHccModuleLaunchKernel/hipExtModuleLaunchKernel, the grid_size is in unit of workitem
    HIP_CALL(hipExtModuleLaunchKernel(gaussian_filer_func, grid_size[0], grid_size[1], grid_size[2],
                                      block_size[0], block_size[1], block_size[2], 0, 0, NULL,
                                      (void **)&config, start, stop));


    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    std::cout << "elapsed time:" << ms << std::endl;

    // check output
    HIP_CALL(hipMemcpy(check_output, output_data_device, height * width * sizeof(float), hipMemcpyDeviceToHost));

    // ptr free
    free(image_data);
    free(filter);
    free(output_data);

    hipFree(image_data_device);
    hipFree(filter_device);
    hipFree(output_data_device);

    return 0;
}