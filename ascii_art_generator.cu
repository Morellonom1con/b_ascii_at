#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define DOWNSAMPLE 4  // Change this to 1, 2, 4, etc. for zoom level

#define BLOCK_SIZE 16
#define CHAR_SET " .,-~:;=!*#$@" // Normal brightness ramp

__global__ void histogramEqualize(unsigned char* img, unsigned char* equalized, int* cdf, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int val = img[idx];
    equalized[idx] = min(max((255.0f * (cdf[val] - cdf[0]) / (float)(size - cdf[0])), 0.0f), 255.0f);
}

__global__ void asciiEdgeKernel(unsigned char* img, char* ascii, int width, int height, int threshold, int dsWidth){
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    if (outX >= width / DOWNSAMPLE || outY >= height / DOWNSAMPLE) return;

    int x = outX * DOWNSAMPLE;
    int y = outY * DOWNSAMPLE;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) return;

    int idx = y * width + x;
    int outIdx = outY * dsWidth + outX;

    // Sobel operators
    int Gx = -img[(y-1)*width + (x-1)] - 2*img[y*width + (x-1)] - img[(y+1)*width + (x-1)]
             + img[(y-1)*width + (x+1)] + 2*img[y*width + (x+1)] + img[(y+1)*width + (x+1)];

    int Gy = -img[(y-1)*width + (x-1)] - 2*img[(y-1)*width + x] - img[(y-1)*width + (x+1)]
             + img[(y+1)*width + (x-1)] + 2*img[(y+1)*width + x] + img[(y+1)*width + (x+1)];

    float magnitude = sqrtf(Gx * Gx + Gy * Gy);
    char ch;

    if (magnitude > threshold) {
        float angle = atan2f((float)Gy, (float)Gx) * 180.0f / 3.14159265f;
        if (angle < 0) angle += 180.0f;

        // Quantize angle
        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) ch = '_';
        else if (angle >= 22.5 && angle < 67.5) ch = '/';
        else if (angle >= 67.5 && angle < 112.5) ch = '|';
        else ch = '\\';
    } else {
        float val = img[idx] / 255.0f;
        int rampIndex = (int)(val * (strlen(CHAR_SET) - 1));
        ch = CHAR_SET[rampIndex];
    }

    ascii[outIdx] = ch;

}

// Host-side histogram equalization helper
void computeCDF(unsigned char* img, int size, int* cdf) {
    int hist[256] = {0};
    for (int i = 0; i < size; i++) hist[img[i]]++;

    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_file.txt>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!img) {
        printf("Image load failed: %s\n", argv[1]);
        return 1;
    }
    int dsWidth = width / DOWNSAMPLE;
    int dsHeight = height / DOWNSAMPLE;
    int asciiSize = dsWidth * dsHeight;

    int size = width * height;
    unsigned char *d_img, *d_eqImg;
    char* d_ascii;
    char* ascii = (char*)malloc(asciiSize);
    int* d_cdf;

    cudaMalloc(&d_img, size);
    cudaMalloc(&d_eqImg, size);
    cudaMalloc(&d_ascii, asciiSize);
    cudaMalloc(&d_cdf, 256 * sizeof(int));

    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    // Histogram Equalization
    int cdf[256];
    computeCDF(img, size, cdf);
    cudaMemcpy(d_cdf, cdf, 256 * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    histogramEqualize<<<blocks, threads>>>(d_img, d_eqImg, d_cdf, size);

    // ASCII Art with edge handling
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((dsWidth + BLOCK_SIZE - 1) / BLOCK_SIZE, (dsHeight + BLOCK_SIZE - 1) / BLOCK_SIZE);
    asciiEdgeKernel<<<gridSize, blockSize>>>(d_eqImg, d_ascii, width, height, 120, dsWidth);

    cudaMemcpy(ascii, d_ascii, asciiSize, cudaMemcpyDeviceToHost);

    FILE* out = fopen(argv[2], "w");
    for (int y = 0; y < dsHeight; y++) {
        for (int x = 0; x < dsWidth; x++) {
            char ch = ascii[y * dsWidth + x];
            fputc(ch, out);
            fputc(ch, out);  // Duplicate character for width
        }
        fputc('\n', out);
    }
    fclose(out);
    
    printf("Saved to %s\n", argv[2]);

    stbi_image_free(img);
    free(ascii);
    cudaFree(d_img);
    cudaFree(d_eqImg);
    cudaFree(d_ascii);
    cudaFree(d_cdf);

    return 0;
}
