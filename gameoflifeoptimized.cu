/**
 * Name: Jeongbin Son
 * Email: json10@crimson.ua.edu
 * Homework #: 5, Optimized
 * @brief This program is for CS 481 - High Performance Computing; HW 5
 * @date 2024, Fall Semester
 * Instructions to compile the program:
 * module load cuda
 * nvcc -o gameoflifeoptimized gameoflifeoptimized.cu
 * Instructions to run the program:
 * ./gameoflifeoptimized 5000 5000 /scratch/ualclsd0197/output_dir
 */

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

#define TILE_SIZE 2

__global__ void gameOfLifeKernel(int *current, int *next, int boardSize) {

    int baseRow = (blockIdx.y * blockDim.y + threadIdx.y) * TILE_SIZE;
    int baseCol = (blockIdx.x * blockDim.x + threadIdx.x) * TILE_SIZE;

    for (int ty = 0; ty < TILE_SIZE; ++ty) {
        int row = baseRow + ty;
        if (row >= boardSize) continue;
        for (int tx = 0; tx < TILE_SIZE; ++tx) {
            int col = baseCol + tx;
            if (col >= boardSize) continue;

            int aliveNeighbors = 0;
            for (int i = -1; i <= 1; ++i) {
                int neighborRow = row + i;
                if (neighborRow < 0 || neighborRow >= boardSize) continue;
                for (int j = -1; j <= 1; ++j) {
                    int neighborCol = col + j;
                    if (neighborCol < 0 || neighborCol >= boardSize) continue;
                    if (i == 0 && j == 0) continue;
                    aliveNeighbors += current[neighborRow * boardSize + neighborCol];
                }
            }

            int index = row * boardSize + col;
            int cell = current[index];
            if (cell == 1) {
                next[index] = (aliveNeighbors < 2 || aliveNeighbors > 3) ? 0 : 1;
            } else {
                next[index] = (aliveNeighbors == 3) ? 1 : 0;
            }
        }
    }
}

// initializing the board but with same random seed as other programs
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

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((boardSize + TILE_SIZE * threadsPerBlock.x - 1) / (TILE_SIZE * threadsPerBlock.x),
                       (boardSize + TILE_SIZE * threadsPerBlock.y - 1) / (TILE_SIZE * threadsPerBlock.y));

    auto start = high_resolution_clock::now();

    for (int gen = 0; gen < generations; ++gen) {
        gameOfLifeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_current, d_next, boardSize);
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