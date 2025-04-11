#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define DOWNSAMPLE 2
#define BLOCK_SIZE 16
#define CHAR_SET " .-:=$@"

// --- CUDA kernel: Histogram Equalization ---
__global__ void histogramEqualize(unsigned char* img, unsigned char* equalized, int* cdf, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    int val = img[idx];
    equalized[idx] = min(max((255.0f * (cdf[val] - cdf[0]) / (float)(size - cdf[0])), 0.0f), 255.0f);
}

// --- CUDA kernel: Gaussian Blur ---
__global__ void gaussianBlur(unsigned char* input, float* output, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int radius = int(2.5f * sigma);
    float sum = 0.0f, weightSum = 0.0f;

    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            int xi = min(max(x + i, 0), width - 1);
            int yj = min(max(y + j, 0), height - 1);
            float dist = i * i + j * j;
            float weight = expf(-dist / (2 * sigma * sigma));
            sum += weight * input[yj * width + xi];
            weightSum += weight;
        }
    }

    output[y * width + x] = sum / weightSum;
}

// --- CUDA kernel: ASCII Edge Detection ---
__global__ void asciiEdgeKernel(float* dogImg, unsigned char* eqImg, char* ascii, int width, int height, int threshold, int dsWidth) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    if (outX >= width / DOWNSAMPLE || outY >= height / DOWNSAMPLE) return;

    int x = outX * DOWNSAMPLE;
    int y = outY * DOWNSAMPLE;
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) return;

    int outIdx = outY * dsWidth + outX;

    // Sobel on DoG image
    float Gx = -dogImg[(y - 1) * width + (x - 1)] - 2 * dogImg[y * width + (x - 1)] - dogImg[(y + 1) * width + (x - 1)]
             + dogImg[(y - 1) * width + (x + 1)] + 2 * dogImg[y * width + (x + 1)] + dogImg[(y + 1) * width + (x + 1)];

    float Gy = -dogImg[(y - 1) * width + (x - 1)] - 2 * dogImg[(y - 1) * width + x] - dogImg[(y - 1) * width + (x + 1)]
             + dogImg[(y + 1) * width + (x - 1)] + 2 * dogImg[(y + 1) * width + x] + dogImg[(y + 1) * width + (x + 1)];

    float magnitude = sqrtf(Gx * Gx + Gy * Gy);
    char ch;

    if (magnitude > threshold) {
        float angle = atan2f(Gy, Gx) * 180.0f / 3.14159265f;
        if (angle < 0) angle += 180.0f;
        if ((angle >= 0 && angle < 30) || (angle >= 150 && angle <= 180)) ch = '_';
        else if (angle >= 30 && angle < 60) ch = '/';
        else if (angle >= 60 && angle < 120) ch = '|';
        else ch = '\\';
    } else {
        float val = eqImg[y * width + x] / 255.0f;
        int rampIndex = (int)(val * (strlen(CHAR_SET) - 1));
        ch = CHAR_SET[rampIndex];
    }

    ascii[outIdx] = ch;
}

// --- Host: CDF computation ---
void computeCDF(unsigned char* img, int size, int* cdf) {
    int hist[256] = {0};
    for (int i = 0; i < size; i++) hist[img[i]]++;
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];
}

__global__ void subtractGaussians(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] - b[i];
    }
}


// --- Main ---
int main(int argc, char* argv[]) {
    cudaEvent_t totalStart, totalStop;
    float totalTime = 0.0f;

    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);
    cudaEventRecord(totalStart);

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

    int size = width * height;
    int dsWidth = width / DOWNSAMPLE;
    int dsHeight = height / DOWNSAMPLE;
    int asciiSize = dsWidth * dsHeight;

    // Device memory
    unsigned char *d_img, *d_eqImg;
    float *d_gauss1, *d_gauss2, *d_dog;
    int* d_cdf;
    char* d_ascii;
    char* ascii = (char*)malloc(asciiSize);

    cudaMalloc(&d_img, size);
    cudaMalloc(&d_eqImg, size);
    cudaMalloc(&d_gauss1, size * sizeof(float));
    cudaMalloc(&d_gauss2, size * sizeof(float));
    cudaMalloc(&d_dog, size * sizeof(float));
    cudaMalloc(&d_cdf, 256 * sizeof(int));
    cudaMalloc(&d_ascii, asciiSize);

    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    // Histogram Equalization
    int cdf[256];
    computeCDF(img, size, cdf);
    cudaMemcpy(d_cdf, cdf, 256 * sizeof(int), cudaMemcpyHostToDevice);
    histogramEqualize<<<(size + 255) / 256, 256>>>(d_img, d_eqImg, d_cdf, size);

    // DoG = Gaussian(small) - Gaussian(large)
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    gaussianBlur<<<gridSize, blockSize>>>(d_eqImg, d_gauss1, width, height, 0.8f);//2.9 for poower ranger
    gaussianBlur<<<gridSize, blockSize>>>(d_eqImg, d_gauss2, width, height, 2.0f);//4.0 for power ranger

    // Subtract the Gaussians to get DoG
    int totalThreads = width * height;
    int threads = 256;
    int blocks = (totalThreads + threads - 1) / threads;
    // quick subtraction kernel
    subtractGaussians<<<blocks, threads>>>(d_gauss1, d_gauss2, d_dog, size);


    // ASCII edge kernel
    asciiEdgeKernel<<<gridSize, blockSize>>>(d_dog, d_eqImg, d_ascii, width, height, 100, dsWidth);//25 for power ranger
    cudaMemcpy(ascii, d_ascii, asciiSize, cudaMemcpyDeviceToHost);

    FILE* out = fopen(argv[2], "w");
    for (int y = 0; y < dsHeight; y++) {
        for (int x = 0; x < dsWidth; x++) {
            char ch = ascii[y * dsWidth + x];
            fputc(ch, out);
            fputc(ch, out);
        }
        fputc('\n', out);
    }
    fclose(out);
    printf("Saved to %s\n", argv[2]);
    cudaEventRecord(totalStop);
    cudaEventSynchronize(totalStop);
    cudaEventElapsedTime(&totalTime, totalStart, totalStop);

    printf("🚀 Total GPU Time: %.3f ms\n", totalTime);

    // Optional cleanup
    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalStop);


    stbi_image_free(img);
    free(ascii);
    cudaFree(d_img);
    cudaFree(d_eqImg);
    cudaFree(d_gauss1);
    cudaFree(d_gauss2);
    cudaFree(d_dog);
    cudaFree(d_ascii);
    cudaFree(d_cdf);
    return 0;
}
