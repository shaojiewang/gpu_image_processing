#pragma once 

#include <stdlib.h>
#include <assert.h>

void* aligned_mem_cpu(std::size_t size, std::size_t alignment)
{
    void* pMemData;
    if(size == 0)
    {
        pMemData = nullptr;
    }
    else
    {
        assert(!(alignment == 0 || (alignment & (alignment - 1))));

        int rtn = posix_memalign(&pMemData, alignment, size);

        assert(rtn == 0);
    }
    return pMemData;
}