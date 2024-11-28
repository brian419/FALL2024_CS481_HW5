// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda.h>
// #include <sys/time.h>

// #define DIES 0
// #define ALIVE 1

// __global__ void compute_life(int *life, int *temp, int n, int iterations, int *flag)
// {
//     extern __shared__ int shared_mem[];
//     int *shared_life = &shared_mem[0];
//     int *shared_temp = &shared_mem[blockDim.x * blockDim.y];

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x * blockDim.x + tx + 1;
//     int by = blockIdx.y * blockDim.y + ty + 1;
//     int idx = by * (n + 2) + bx;

//     shared_life[ty * blockDim.x + tx] = life[idx];
//     __syncthreads();

//     for (int k = 0; k < iterations; k++)
//     {
//         int value = 0;

//         if (tx > 0 && tx < blockDim.x - 1 && ty > 0 && ty < blockDim.y - 1)
//         {
//             value = shared_life[(ty - 1) * blockDim.x + (tx - 1)] +
//                     shared_life[(ty - 1) * blockDim.x + tx] +
//                     shared_life[(ty - 1) * blockDim.x + (tx + 1)] +
//                     shared_life[ty * blockDim.x + (tx - 1)] +
//                     shared_life[ty * blockDim.x + (tx + 1)] +
//                     shared_life[(ty + 1) * blockDim.x + (tx - 1)] +
//                     shared_life[(ty + 1) * blockDim.x + tx] +
//                     shared_life[(ty + 1) * blockDim.x + (tx + 1)];
//         }

//         if (shared_life[ty * blockDim.x + tx])
//         {
//             if (value < 2 || value > 3)
//             {
//                 shared_temp[ty * blockDim.x + tx] = DIES;
//                 atomicAdd(flag, 1);
//             }
//             else
//             {
//                 shared_temp[ty * blockDim.x + tx] = ALIVE;
//             }
//         }
//         else
//         {
//             if (value == 3)
//             {
//                 shared_temp[ty * blockDim.x + tx] = ALIVE;
//                 atomicAdd(flag, 1);
//             }
//             else
//             {
//                 shared_temp[ty * blockDim.x + tx] = DIES;
//             }
//         }
//         __syncthreads();

//         int *tmp = shared_life;
//         shared_life = shared_temp;
//         shared_temp = tmp;

//         __syncthreads();
//     }

//     life[idx] = shared_life[ty * blockDim.x + tx];
// }

// double gettime(void)
// {
//     struct timeval tval;
//     gettimeofday(&tval, NULL);
//     return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
// }

// void initialize_life(int *life, int n)
// {
//     for (int i = 0; i < (n + 2) * (n + 2); i++)
//     {
//         life[i] = (rand() % 2);
//     }
// }

// int main(int argc, char **argv)
// {
//     if (argc != 3)
//     {
//         printf("Usage: %s <grid size> <iterations>\n", argv[0]);
//         exit(1);
//     }

//     int n = atoi(argv[1]);
//     int iterations = atoi(argv[2]);
//     int *h_life;
//     int *d_life, *d_temp, *d_flag;
//     int block_size = 16;
//     double start, end;

//     h_life = (int *)malloc((n + 2) * (n + 2) * sizeof(int));
//     initialize_life(h_life, n);

//     cudaError_t err = cudaMalloc(&d_life, (n + 2) * (n + 2) * sizeof(int));
//     if (err != cudaSuccess)
//     {
//         printf("CUDA Malloc Error for d_life: %s\n", cudaGetErrorString(err));
//     }
//     // cudaMalloc(&d_life, (n + 2) * (n + 2) * sizeof(int));
//     cudaMalloc(&d_temp, (n + 2) * (n + 2) * sizeof(int));
//     cudaMalloc(&d_flag, sizeof(int));

//     cudaMemcpy(d_life, h_life, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyHostToDevice);

//     // verify if the output grid is modified
//     int changes = 0;
//     for (int i = 1; i <= n; i++)
//     {
//         for (int j = 1; j <= n; j++)
//         {
//             if (h_life[i * (n + 2) + j] != 0)
//             {
//                 changes++;
//             }
//         }
//     }
//     printf("Number of cells modified: %d\n", changes);

//     dim3 dim_block(block_size, block_size);
//     dim3 dim_grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

//     start = gettime();

//     int host_flag = 0;
//     cudaMemcpy(d_flag, &host_flag, sizeof(int), cudaMemcpyHostToDevice);

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





// v2

// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda.h>
// #include <sys/time.h>
// #include <string>
// #include <fstream>

// #define DIES 0
// #define ALIVE 1

// using namespace std;

// __global__ void compute_life(int *life, int *temp, int n, int iterations, int *flag)
// {
//     extern __shared__ int shared_mem[];
//     int *shared_life = &shared_mem[0];
//     int *shared_temp = &shared_mem[blockDim.x * blockDim.y];

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x * blockDim.x + tx + 1;
//     int by = blockIdx.y * blockDim.y + ty + 1;
//     int idx = by * (n + 2) + bx;

//     shared_life[ty * blockDim.x + tx] = life[idx];
//     __syncthreads();

//     for (int k = 0; k < iterations; k++)
//     {
//         int value = 0;

//         if (tx > 0 && tx < blockDim.x - 1 && ty > 0 && ty < blockDim.y - 1)
//         {
//             value = shared_life[(ty - 1) * blockDim.x + (tx - 1)] +
//                     shared_life[(ty - 1) * blockDim.x + tx] +
//                     shared_life[(ty - 1) * blockDim.x + (tx + 1)] +
//                     shared_life[ty * blockDim.x + (tx - 1)] +
//                     shared_life[ty * blockDim.x + (tx + 1)] +
//                     shared_life[(ty + 1) * blockDim.x + (tx - 1)] +
//                     shared_life[(ty + 1) * blockDim.x + tx] +
//                     shared_life[(ty + 1) * blockDim.x + (tx + 1)];
//         }

//         if (shared_life[ty * blockDim.x + tx])
//         {
//             if (value < 2 || value > 3)
//             {
//                 shared_temp[ty * blockDim.x + tx] = DIES;
//                 atomicAdd(flag, 1);
//             }
//             else
//             {
//                 shared_temp[ty * blockDim.x + tx] = ALIVE;
//             }
//         }
//         else
//         {
//             if (value == 3)
//             {
//                 shared_temp[ty * blockDim.x + tx] = ALIVE;
//                 atomicAdd(flag, 1);
//             }
//             else
//             {
//                 shared_temp[ty * blockDim.x + tx] = DIES;
//             }
//         }
//         __syncthreads();

//         int *tmp = shared_life;
//         shared_life = shared_temp;
//         shared_temp = tmp;

//         __syncthreads();
//     }

//     life[idx] = shared_life[ty * blockDim.x + tx];
// }

// double gettime(void)
// {
//     struct timeval tval;
//     gettimeofday(&tval, NULL);
//     return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
// }

// void initialize_life(int *life, int n)
// {
//     srand(12345);
//     for (int i = 0; i < (n + 2) * (n + 2); i++)
//     {
//         life[i] = (rand() % 2);
//     }
// }

// void writeFinalBoardToFile(const int *board, int n, int iterations, const string &outputDir)
// {
//     string correctedOutputDir = outputDir;
//     if (outputDir.back() != '/')
//     {
//         correctedOutputDir += "/";
//     }

//     string fileName = correctedOutputDir + "hw5_GPU_" + to_string(n) + "x" + to_string(n) +
//                       "_board_" + to_string(iterations) + "_iterations_testcase.txt";

//     ofstream outFile(fileName);

//     if (!outFile)
//     {
//         printf("Error creating output file: %s\n", fileName.c_str());
//         return;
//     }

//     for (int i = 1; i <= n; ++i)
//     { 
//         for (int j = 1; j <= n; ++j)
//         {
//             outFile << (board[i * (n + 2) + j] ? '*' : '.') << " ";
//         }
//         outFile << endl;
//     }

//     outFile.close();
//     printf("Final board written to %s\n", fileName.c_str());
// }


// int main(int argc, char **argv)
// {
//     if (argc != 4)
//     {
//         printf("Usage: %s <grid size> <iterations> <output directory>\n", argv[0]);
//         exit(1);
//     }

//     int n = atoi(argv[1]);
//     int iterations = atoi(argv[2]);
//     string outputDir = argv[3];

//     int *h_life;
//     int *d_life, *d_temp, *d_flag;
//     int block_size = 16;
//     double start, end;

//     h_life = (int *)malloc((n + 2) * (n + 2) * sizeof(int));
//     initialize_life(h_life, n);

//     cudaError_t err = cudaMalloc(&d_life, (n + 2) * (n + 2) * sizeof(int));
//     if (err != cudaSuccess)
//     {
//         printf("CUDA Malloc Error for d_life: %s\n", cudaGetErrorString(err));
//     }

//     cudaMalloc(&d_temp, (n + 2) * (n + 2) * sizeof(int));
//     cudaMalloc(&d_flag, sizeof(int));

//     cudaMemcpy(d_life, h_life, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyHostToDevice);

//     dim3 dim_block(block_size, block_size);
//     dim3 dim_grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

//     start = gettime();

//     int host_flag = 0;
//     cudaMemcpy(d_flag, &host_flag, sizeof(int), cudaMemcpyHostToDevice);

//     compute_life<<<dim_grid, dim_block, 2 * block_size * block_size * sizeof(int)>>>(d_life, d_temp, n, iterations, d_flag);
//     cudaDeviceSynchronize();

//     end = gettime();

//     cudaMemcpy(h_life, d_life, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyDeviceToHost);

//     writeFinalBoardToFile(h_life, n, iterations, outputDir);

//     printf("Time taken for %d iterations: %f seconds\n", iterations, end - start);

//     free(h_life);

//     cudaFree(d_life);
//     cudaFree(d_temp);
//     cudaFree(d_flag);

//     return 0;
// }







// v3
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
        exit(1); \
    }

__global__ void gameOfLifeKernel(int *current, int *next, int boardSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < boardSize && col < boardSize) {
        int aliveNeighbors = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0) continue;
                int newRow = row + i;
                int newCol = col + j;
                if (newRow >= 0 && newRow < boardSize && newCol >= 0 && newCol < boardSize) {
                    aliveNeighbors += current[newRow * boardSize + newCol];
                }
            }
        }
        int index = row * boardSize + col;
        next[index] = (current[index] == 1) ? (aliveNeighbors < 2 || aliveNeighbors > 3 ? 0 : 1) : (aliveNeighbors == 3 ? 1 : 0);
    }
}

void initializeBoard(int *board, int boardSize) {
    srand(12345); 
    for (int i = 0; i < boardSize * boardSize; ++i) {
        board[i] = rand() % 2;
    }
}

// final board to file
void writeFinalBoardToFile(const int *board, int n, int iterations, const string &outputDir)
{
    string correctedOutputDir = outputDir;
    if (outputDir.back() != '/')
    {
        correctedOutputDir += "/";
    }

    string fileName = correctedOutputDir + "hw5_GPU_" + to_string(n) + "x" + to_string(n) +
                      "_board_" + to_string(iterations) + "_iterations_testcase.txt";

    ofstream outFile(fileName);

    if (!outFile)
    {
        printf("Error creating output file: %s\n", fileName.c_str());
        return;
    }

    for (int i = 0; i < n; ++i)
    { 
        for (int j = 0; j < n; ++j)
        {
            outFile << (board[i * n + j] ? '*' : '.') << " ";
        }
        outFile << endl;
    }

    outFile.close();
    printf("Final board written to %s\n", fileName.c_str());
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <board size> <generations>" << endl;
        return 1;
    }



    int boardSize = stoi(argv[1]);
    int generations = stoi(argv[2]);
    string outputDir = argv[3];

    size_t size = boardSize * boardSize * sizeof(int);
    int *h_current = new int[boardSize * boardSize];
    int *h_next = new int[boardSize * boardSize];

    initializeBoard(h_current, boardSize);

    int *d_current, *d_next;
    CHECK_CUDA_ERROR(cudaMalloc(&d_current, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_next, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_current, h_current, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((boardSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (boardSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    auto start = high_resolution_clock::now();

    for (int gen = 0; gen < generations; ++gen) {
        gameOfLifeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_current, d_next, boardSize);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaMemcpy(d_current, d_next, size, cudaMemcpyDeviceToDevice));
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);


    writeFinalBoardToFile(h_next, boardSize, generations, outputDir);

    CHECK_CUDA_ERROR(cudaMemcpy(h_next, d_current, size, cudaMemcpyDeviceToHost));
    cout << "Simulation completed in " << duration.count() << " ms." << endl;

    cudaFree(d_current);
    cudaFree(d_next);
    delete[] h_current;
    delete[] h_next;

    return 0;
}
