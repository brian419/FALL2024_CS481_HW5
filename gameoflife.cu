// // game of life for HW5 GPU VERSION

// // going to test code on cluster as I don't have NVDIA GPU but starting with a simple addition until number 10 loop

// // Instructions to compile the program: nvcc -o gameoflife gameoflife.cu

// // Instructions to run the program: ./gameoflife

// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda.h>
// #include <sys/time.h>

// #define DIES 0
// #define ALIVE 1

// __global__ void compute_life(int* life, int* temp, int n, int iterations, int* flag) {
//     extern __shared__ int shared_mem[];
//     int* shared_life = &shared_mem[0];
//     int* shared_temp = &shared_mem[blockDim.x * blockDim.y];

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x * blockDim.x + tx + 1;
//     int by = blockIdx.y * blockDim.y + ty + 1;
//     int idx = by * (n + 2) + bx;

//     shared_life[ty * blockDim.x + tx] = life[idx];
//     __syncthreads();

//     for (int k = 0; k < iterations; k++) {
//         int value = 0;
//         if (tx > 0 && tx < blockDim.x - 1 && ty > 0 && ty < blockDim.y - 1) {
//             value = shared_life[(ty - 1) * blockDim.x + (tx - 1)] +
//                     shared_life[(ty - 1) * blockDim.x + tx] +
//                     shared_life[(ty - 1) * blockDim.x + (tx + 1)] +
//                     shared_life[ty * blockDim.x + (tx - 1)] +
//                     shared_life[ty * blockDim.x + (tx + 1)] +
//                     shared_life[(ty + 1) * blockDim.x + (tx - 1)] +
//                     shared_life[(ty + 1) * blockDim.x + tx] +
//                     shared_life[(ty + 1) * blockDim.x + (tx + 1)];
//         }

//         if (shared_life[ty * blockDim.x + tx]) {
//             if (value < 2 || value > 3) {
//                 shared_temp[ty * blockDim.x + tx] = DIES;
//                 atomicAdd(flag, 1);
//             } else {
//                 shared_temp[ty * blockDim.x + tx] = ALIVE;
//             }
//         } else {
//             if (value == 3) {
//                 shared_temp[ty * blockDim.x + tx] = ALIVE;
//                 atomicAdd(flag, 1);
//             } else {
//                 shared_temp[ty * blockDim.x + tx] = DIES;
//             }
//         }
//         __syncthreads();

//         int* tmp = shared_life;
//         shared_life = shared_temp;
//         shared_temp = tmp;
//     }

//     life[idx] = shared_life[ty * blockDim.x + tx];
// }

// double gettime(void) {
//     struct timeval tval;
//     gettimeofday(&tval, NULL);
//     return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
// }

// void initialize_life(int* life, int n) {
//     for (int i = 0; i < n * n; i++) {
//         life[i] = (rand() % 2);
//     }
// }

// int main(int argc, char** argv) {
//     int n = atoi(argv[1]);
//     int iterations = atoi(argv[2]);
//     int* h_life;
//     int* d_life, *d_temp, *d_flag;
//     int block_size = 16;
//     double start, end;

//     h_life = (int*)malloc((n + 2) * (n + 2) * sizeof(int));
//     initialize_life(h_life, n);

//     cudaMalloc(&d_life, (n + 2) * (n + 2) * sizeof(int));
//     cudaMalloc(&d_temp, (n + 2) * (n + 2) * sizeof(int));
//     cudaMalloc(&d_flag, sizeof(int));

//     cudaMemcpy(d_life, h_life, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyHostToDevice);

//     dim3 dim_block(block_size, block_size);
//     dim3 dim_grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

//     start = gettime();
//     compute_life<<<dim_grid, dim_block, 2 * block_size * block_size * sizeof(int)>>>(d_life, d_temp, n, iterations, d_flag);
//     cudaDeviceSynchronize();
//     end = gettime();

//     cudaMemcpy(h_life, d_life, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyDeviceToHost);

//     printf("Time taken for %d iterations: %f seconds\n", iterations, end - start);

//     free(h_life);
//     cudaFree(d_life);
//     cudaFree(d_temp);
//     cudaFree(d_flag);

//     return 0;
// }

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

#define DIES 0
#define ALIVE 1

__global__ void compute_life(int *life, int *temp, int n, int iterations, int *flag)
{
    extern __shared__ int shared_mem[];
    int *shared_life = &shared_mem[0];
    int *shared_temp = &shared_mem[blockDim.x * blockDim.y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x + tx + 1;
    int by = blockIdx.y * blockDim.y + ty + 1;
    int idx = by * (n + 2) + bx;

    shared_life[ty * blockDim.x + tx] = life[idx];
    __syncthreads();

    for (int k = 0; k < iterations; k++)
    {
        int value = 0;

        if (tx > 0 && tx < blockDim.x - 1 && ty > 0 && ty < blockDim.y - 1)
        {
            value = shared_life[(ty - 1) * blockDim.x + (tx - 1)] +
                    shared_life[(ty - 1) * blockDim.x + tx] +
                    shared_life[(ty - 1) * blockDim.x + (tx + 1)] +
                    shared_life[ty * blockDim.x + (tx - 1)] +
                    shared_life[ty * blockDim.x + (tx + 1)] +
                    shared_life[(ty + 1) * blockDim.x + (tx - 1)] +
                    shared_life[(ty + 1) * blockDim.x + tx] +
                    shared_life[(ty + 1) * blockDim.x + (tx + 1)];
        }

        if (shared_life[ty * blockDim.x + tx])
        {
            if (value < 2 || value > 3)
            {
                shared_temp[ty * blockDim.x + tx] = DIES;
                atomicAdd(flag, 1);
            }
            else
            {
                shared_temp[ty * blockDim.x + tx] = ALIVE;
            }
        }
        else
        {
            if (value == 3)
            {
                shared_temp[ty * blockDim.x + tx] = ALIVE;
                atomicAdd(flag, 1);
            }
            else
            {
                shared_temp[ty * blockDim.x + tx] = DIES;
            }
        }
        __syncthreads();

        int *tmp = shared_life;
        shared_life = shared_temp;
        shared_temp = tmp;

        __syncthreads();
    }

    life[idx] = shared_life[ty * blockDim.x + tx];
}

double gettime(void)
{
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
}

void initialize_life(int *life, int n)
{
    for (int i = 0; i < (n + 2) * (n + 2); i++)
    {
        life[i] = (rand() % 2);
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <grid size> <iterations>\n", argv[0]);
        exit(1);
    }

    int n = atoi(argv[1]);
    int iterations = atoi(argv[2]);
    int *h_life;
    int *d_life, *d_temp, *d_flag;
    int block_size = 16;
    double start, end;

    h_life = (int *)malloc((n + 2) * (n + 2) * sizeof(int));
    initialize_life(h_life, n);

    cudaError_t err = cudaMalloc(&d_life, (n + 2) * (n + 2) * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("CUDA Malloc Error for d_life: %s\n", cudaGetErrorString(err));
    }
    // cudaMalloc(&d_life, (n + 2) * (n + 2) * sizeof(int));
    cudaMalloc(&d_temp, (n + 2) * (n + 2) * sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));

    cudaMemcpy(d_life, h_life, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyHostToDevice);

    // verify if the output grid is modified
    int changes = 0;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            if (h_life[i * (n + 2) + j] != 0)
            {
                changes++;
            }
        }
    }
    printf("Number of cells modified: %d\n", changes);

    dim3 dim_block(block_size, block_size);
    dim3 dim_grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    start = gettime();

    int host_flag = 0;
    cudaMemcpy(d_flag, &host_flag, sizeof(int), cudaMemcpyHostToDevice);

    compute_life<<<dim_grid, dim_block, 2 * block_size * block_size * sizeof(int)>>>(d_life, d_temp, n, iterations, d_flag);
    cudaDeviceSynchronize();

    end = gettime();

    cudaMemcpy(h_life, d_life, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Time taken for %d iterations: %f seconds\n", iterations, end - start);

    free(h_life);
    cudaFree(d_life);
    cudaFree(d_temp);
    cudaFree(d_flag);

    return 0;
}
