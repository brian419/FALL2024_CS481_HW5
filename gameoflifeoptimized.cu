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
__global__ void gameOfLifeInnerKernel(const int* current, int* next, int boardSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (row < boardSize -1 && col < boardSize -1) {
        int index = row * boardSize + col;
        int aliveNeighbors =
            current[(row - 1) * boardSize + (col - 1)] +
            current[(row - 1) * boardSize + col] +
            current[(row - 1) * boardSize + (col + 1)] +
            current[row * boardSize + (col - 1)] +
            current[row * boardSize + (col + 1)] +
            current[(row + 1) * boardSize + (col - 1)] +
            current[(row + 1) * boardSize + col] +
            current[(row + 1) * boardSize + (col + 1)];
        int cellState = current[index];
        next[index] = (cellState == 1) ? ((aliveNeighbors < 2 || aliveNeighbors > 3) ? 0 : 1) : (aliveNeighbors == 3 ? 1 : 0);
    }
}
__global__ void gameOfLifeBoundaryKernel(const int* current, int* next, int boardSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalBoundaryCells = boardSize * 4 - 4;
    if (tid < totalBoundaryCells) {
        int row, col;
        if (tid < boardSize) {
            row = 0;
            col = tid;
        } else if (tid < 2 * boardSize - 1) {
            row = boardSize -1;
            col = tid - boardSize;
        } else if (tid < 2 * boardSize - 1 + boardSize -2) {
            row = tid - (2 * boardSize -1) + 1;
            col = 0;
        } else {
            row = tid - (2 * boardSize -1 + boardSize -2) + 1;
            col = boardSize -1;
        }
        int aliveNeighbors = 0;
        for (int i = -1; i <=1; ++i) {
            for (int j = -1; j <=1; ++j) {
                if (i == 0 && j == 0) continue;
                int newRow = row + i;
                int newCol = col + j;
                if (newRow >= 0 && newRow < boardSize && newCol >= 0 && newCol < boardSize) {
                    aliveNeighbors += current[newRow * boardSize + newCol];
                }
            }
        }
        int index = row * boardSize + col;
        int cellState = current[index];
        next[index] = (cellState == 1) ? ((aliveNeighbors < 2 || aliveNeighbors > 3) ? 0 : 1) : (aliveNeighbors == 3 ? 1 : 0);
    }
}
void initializeBoard(int *board, int boardSize) {
    srand(12345);
    for (int i = 0; i < boardSize * boardSize; ++i) {
        board[i] = rand() % 2;
    }
}
void writeFinalBoardToFile(const int *board, int n, int iterations, const string &outputDir)
{
    string correctedOutputDir = outputDir;
    if (outputDir.back() != '/')
    {
        correctedOutputDir += "/";
    }
    string fileName = correctedOutputDir + "hw5_GPU_" + to_string(n) + "x" + to_string(n) +
                    "_board_" + to_string(iterations) + "_iterations_V3code_OptimizedV2_testcase.txt";
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
    int innerBoardSize = boardSize - 2;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGridInner((innerBoardSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                            (innerBoardSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int totalBoundaryCells = boardSize * 4 - 4;
    int threadsPerBlockBoundary = 256;
    int blocksPerGridBoundary = (totalBoundaryCells + threadsPerBlockBoundary -1)/threadsPerBlockBoundary;
    auto start = high_resolution_clock::now();
    for (int gen = 0; gen < generations; ++gen) {
        gameOfLifeInnerKernel<<<blocksPerGridInner, threadsPerBlock>>>(d_current, d_next, boardSize);
        gameOfLifeBoundaryKernel<<<blocksPerGridBoundary, threadsPerBlockBoundary>>>(d_current, d_next, boardSize);
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
