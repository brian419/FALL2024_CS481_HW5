// game of life for HW5 GPU VERSION

// going to test code on cluster as I don't have NVDIA GPU but starting with a simple addition until number 10 loop

// Instructions to compile the program: nvcc -o gameoflife gameoflife.cu

// Instructions to run the program: ./gameoflife

#include <iostream>
#include <cuda_runtime.h>

// kernel function to count to 10
__global__ void count_to_ten() {
    int thread_id = threadIdx.x; // get the thread index

    // only let the first thread in the block count to 10
    if (thread_id == 0) {
        for (int i = 1; i <= 10; i++) {
            printf("Count: %d\n", i);
        }
    }
}

int main() {
    std::cout << "Starting the GPU count to 10 program..." << std::endl;

    // launch the kernel with 1 block and 1 thread
    count_to_ten<<<1, 1>>>();

    // synchronize GPU and CPU to ensure all output is printed
    cudaDeviceSynchronize();

    std::cout << "Finished counting on GPU." << std::endl;

    return 0;
}
