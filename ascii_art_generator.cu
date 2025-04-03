#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define BLOCK_SIZE 8
#define CHAR_SET " .,-~:;=!*#$@"

__global__ void generateAsciiArt(unsigned char* input, char* asciiOutput, int width, int height, int asciiWidth, int asciiHeight) {
    int blockX = blockIdx.x * BLOCK_SIZE;
    int blockY = blockIdx.y * BLOCK_SIZE;
    
    if (blockX >= width || blockY >= height) return;
    
    float sum = 0.0f;

    for (int y = blockY; y < blockY + BLOCK_SIZE && y < height; y++) {
        for (int x = blockX; x < blockX + BLOCK_SIZE && x < width; x++) {
            sum += input[y * width + x];
        }
    }

    float avg = sum / (BLOCK_SIZE * BLOCK_SIZE);
    int charIndex = (avg / 255.0f) * (strlen(CHAR_SET) - 1);
    
    asciiOutput[(blockY / BLOCK_SIZE) * asciiWidth + (blockX / BLOCK_SIZE)] = CHAR_SET[charIndex];
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_file.txt>" << std::endl;
        return EXIT_FAILURE;
    }

    int width, height, channels;
    unsigned char* image = stbi_load(argv[1], &width, &height, &channels, 1);

    if (!image) {
        std::cerr << "Error: Could not load image " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    int asciiWidth = width / BLOCK_SIZE;
    int asciiHeight = height / BLOCK_SIZE;

    unsigned char* d_image;
    char* d_asciiOutput;

    cudaMalloc(&d_image, width * height);
    cudaMalloc(&d_asciiOutput, asciiWidth * asciiHeight);

    cudaMemcpy(d_image, image, width * height, cudaMemcpyHostToDevice);

    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    generateAsciiArt<<<blocks, threads>>>(d_image, d_asciiOutput, width, height, asciiWidth, asciiHeight);

    std::vector<char> asciiOutput(asciiWidth * asciiHeight);
    cudaMemcpy(asciiOutput.data(), d_asciiOutput, asciiWidth * asciiHeight, cudaMemcpyDeviceToHost);

    std::ofstream outFile(argv[2]);
    if (!outFile) {
        std::cerr << "Error: Could not open output file " << argv[2] << std::endl;
        return EXIT_FAILURE;
    }

    for (int y = 0; y < asciiHeight; y++) {
        outFile.write(&asciiOutput[y * asciiWidth], asciiWidth);
        outFile << '\n';
    }

    outFile.close();
    std::cout << "ASCII art saved to " << argv[2] << std::endl;

    stbi_image_free(image);
    cudaFree(d_image);
    cudaFree(d_asciiOutput);

    return EXIT_SUCCESS;
}
