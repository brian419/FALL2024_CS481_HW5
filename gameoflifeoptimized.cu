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
    extern __shared__ int sharedBoard[];
    int localRow = threadIdx.y + 1;
    int localCol = threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int globalIdx = row * boardSize + col;
    int bitIdx = globalIdx % 32;
    int arrayIdx = globalIdx / 32;

    if (row < boardSize && col < boardSize) {
        sharedBoard[localRow * (blockDim.x + 2) + localCol] = (current[arrayIdx] >> bitIdx) & 1;

        if (threadIdx.x == 0 && col > 0)
            sharedBoard[localRow * (blockDim.x + 2)] = (current[(arrayIdx - 1)] >> (31)) & 1;
        if (threadIdx.x == blockDim.x - 1 && col < boardSize - 1)
            sharedBoard[localRow * (blockDim.x + 2) + blockDim.x + 1] = (current[(arrayIdx + 1)] & 1);
        if (threadIdx.y == 0 && row > 0)
            sharedBoard[(localRow - 1) * (blockDim.x + 2) + localCol] = (current[arrayIdx - boardSize / 32] >> bitIdx) & 1;
        if (threadIdx.y == blockDim.y - 1 && row < boardSize - 1)
            sharedBoard[(localRow + 1) * (blockDim.x + 2) + localCol] = (current[arrayIdx + boardSize / 32] >> bitIdx) & 1;

        __syncthreads();

        int aliveNeighbors = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0) continue;
                aliveNeighbors += sharedBoard[(localRow + i) * (blockDim.x + 2) + (localCol + j)];
            }
        }

        int currentCellState = sharedBoard[localRow * (blockDim.x + 2) + localCol];
        int newState = (currentCellState == 1)
                           ? (aliveNeighbors < 2 || aliveNeighbors > 3 ? 0 : 1)
                           : (aliveNeighbors == 3 ? 1 : 0);

        atomicOr(&next[arrayIdx], newState << bitIdx);
    }
}

void initializeBoardBitPacked(int *board, int boardSize) {
    srand(12345);
    for (int i = 0; i < boardSize * boardSize; ++i) {
        int bitIndex = i % 32;
        int arrayIndex = i / 32;
        if (bitIndex == 0) board[arrayIndex] = 0;
        board[arrayIndex] |= ((rand() % 2) << bitIndex);
    }
}

void writeFinalBoardToFile(const int *board, int n, int iterations, const string &outputDir) {
    string correctedOutputDir = outputDir;
    if (outputDir.back() != '/')
        correctedOutputDir += "/";
    string fileName = correctedOutputDir + "hw5_GPU_" + to_string(n) + "x" + to_string(n) +
                      "_board_" + to_string(iterations) + "_iterations_V3code_Optimized_testcase.txt";
    ofstream outFile(fileName);
    if (!outFile)
        return;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int globalIdx = i * n + j;
            int bitIdx = globalIdx % 32;
            int arrayIdx = globalIdx / 32;
            outFile << (((board[arrayIdx] >> bitIdx) & 1) ? '*' : '.') << " ";
        }
        outFile << endl;
    }
    outFile.close();
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " <board size> <generations> <output directory>" << endl;
        return 1;
    }

    int boardSize = stoi(argv[1]);
    int generations = stoi(argv[2]);
    string outputDir = argv[3];

    size_t numPackedInts = (boardSize * boardSize + 31) / 32;
    size_t size = numPackedInts * sizeof(int);
    int *h_current = new int[numPackedInts];
    int *h_next = new int[numPackedInts];

    initializeBoardBitPacked(h_current, boardSize);

    int *d_current, *d_next;
    CHECK_CUDA_ERROR(cudaMalloc(&d_current, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_next, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_current, h_current, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((boardSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (boardSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t sharedMemorySize = (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * sizeof(int);

    auto start = high_resolution_clock::now();

    for (int gen = 0; gen < generations; ++gen) {
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
