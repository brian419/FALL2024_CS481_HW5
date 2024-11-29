// v3
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;

#define CHECK_CUDA_ERROR(err) \
    { \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    }

__global__ void gameOfLifeKernelOptimized(int *current, int *next, int boardSize) {
    extern __shared__ int shared[];
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < boardSize && col < boardSize) {
        shared[threadRow * (blockDim.x + 2) + (threadCol + 1)] = current[row * boardSize + col];
        if (threadCol == 0 && col > 0)
            shared[threadRow * (blockDim.x + 2)] = current[row * boardSize + col - 1];
        if (threadCol == blockDim.x - 1 && col < boardSize - 1)
            shared[threadRow * (blockDim.x + 2) + (blockDim.x + 1)] = current[row * boardSize + col + 1];
    }

    __syncthreads();

    if (row < boardSize && col < boardSize) {
        int aliveNeighbors = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0) continue;
                int localRow = threadRow + i;
                int localCol = threadCol + j;
                if (localRow >= 0 && localRow < blockDim.y && localCol >= 0 && localCol < blockDim.x) {
                    aliveNeighbors += shared[localRow * (blockDim.x + 2) + localCol];
                }
            }
        }
        int index = row * boardSize + col;
        next[index] = (shared[threadRow * (blockDim.x + 2) + threadCol] == 1)
                          ? (aliveNeighbors < 2 || aliveNeighbors > 3 ? 0 : 1)
                          : (aliveNeighbors == 3 ? 1 : 0);
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
                      "_board_" + to_string(iterations) + "_iterations_V3code_Optimized_testcase.txt";

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
    if (argc != 4) { 
        cout << "Usage: " << argv[0] << " <board size> <generations> <output directory>" << endl;
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
        // gameOfLifeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_current, d_next, boardSize);
        size_t sharedMemorySize = (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * sizeof(int);
        gameOfLifeKernelOptimized<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(d_current, d_next, boardSize);

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaMemcpy(d_current, d_next, size, cudaMemcpyDeviceToDevice));
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    CHECK_CUDA_ERROR(cudaMemcpy(h_next, d_current, size, cudaMemcpyDeviceToHost));

    writeFinalBoardToFile(h_next, boardSize, generations, outputDir);

    cout << "Simulation completed in " << duration.count() << " ms." << endl;

    cudaFree(d_current);
    cudaFree(d_next);
    delete[] h_current;
    delete[] h_next;

    return 0;
}
