#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define DOWNSAMPLE 2
#define CHAR_SET " .,-~:;=!*#$@"

void computeCDF(unsigned char* img, int size, int* cdf) {
    int hist[256] = {0};
    for (int i = 0; i < size; i++) hist[img[i]]++;

    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }
}

void histogramEqualizeCPU(unsigned char* img, unsigned char* equalized, int* cdf, int size) {
    for (int i = 0; i < size; i++) {
        int val = img[i];
        equalized[i] = fmin(fmax((255.0f * (cdf[val] - cdf[0]) / (float)(size - cdf[0])), 0.0f), 255.0f);
    }
}

void gaussianBlurCPU(unsigned char* input, float* output, int width, int height, float sigma) {
    int radius = (int)(2.5f * sigma);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            float weightSum = 0.0f;
            
            for (int j = -radius; j <= radius; j++) {
                for (int i = -radius; i <= radius; i++) {
                    int xi = fmin(fmax(x + i, 0), width - 1);
                    int yj = fmin(fmax(y + j, 0), height - 1);
                    float dist = i * i + j * j;
                    float weight = expf(-dist / (2 * sigma * sigma));
                    sum += weight * input[yj * width + xi];
                    weightSum += weight;
                }
            }
            
            output[y * width + x] = sum / weightSum;
        }
    }
}

void subtractGaussiansCPU(float* a, float* b, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] - b[i];
    }
}

void asciiEdgeCPU(float* dogImg, unsigned char* eqImg, char* ascii, int width, int height, int threshold, int dsWidth, int dsHeight) {
    for (int outY = 0; outY < dsHeight; outY++) {
        for (int outX = 0; outX < dsWidth; outX++) {
            int x = outX * DOWNSAMPLE;
            int y = outY * DOWNSAMPLE;

            if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) continue;

            int outIdx = outY * dsWidth + outX;

            // Sobel on DoG image (same as GPU version)
            float Gx = -dogImg[(y - 1) * width + (x - 1)] - 2 * dogImg[y * width + (x - 1)] - dogImg[(y + 1) * width + (x - 1)]
                      + dogImg[(y - 1) * width + (x + 1)] + 2 * dogImg[y * width + (x + 1)] + dogImg[(y + 1) * width + (x + 1)];

            float Gy = -dogImg[(y - 1) * width + (x - 1)] - 2 * dogImg[(y - 1) * width + x] - dogImg[(y - 1) * width + (x + 1)]
                      + dogImg[(y + 1) * width + (x - 1)] + 2 * dogImg[(y + 1) * width + x] + dogImg[(y + 1) * width + (x + 1)];

            float magnitude = sqrtf(Gx * Gx + Gy * Gy);
            char ch;

            if (magnitude > threshold) {
                float angle = atan2f(Gy, Gx) * 180.0f / 3.14159265f;
                if (angle < 0) angle += 180.0f;
                if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) ch = '_';
                else if (angle >= 22.5 && angle < 67.5) ch = '/';
                else if (angle >= 67.5 && angle < 112.5) ch = '|';
                else ch = '\\';
            } else {
                float val = eqImg[y * width + x] / 255.0f;
                int rampIndex = (int)(val * (strlen(CHAR_SET) - 1));
                ch = CHAR_SET[rampIndex];
            }

            ascii[outIdx] = ch;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_file.txt>\n", argv[0]);
        return 1;
    }

    clock_t start = clock();

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

    unsigned char* equalized = (unsigned char*)malloc(size);
    float* gauss1 = (float*)malloc(size * sizeof(float));
    float* gauss2 = (float*)malloc(size * sizeof(float));
    float* dog = (float*)malloc(size * sizeof(float));
    char* ascii = (char*)malloc(asciiSize);
    int cdf[256];

    // Histogram Equalization (same as before)
    computeCDF(img, size, cdf);
    histogramEqualizeCPU(img, equalized, cdf, size);
    
    // DoG = Gaussian(small) - Gaussian(large) (matching GPU version)
    gaussianBlurCPU(equalized, gauss1, width, height, 0.8f);
    gaussianBlurCPU(equalized, gauss2, width, height, 2.0f);
    subtractGaussiansCPU(gauss1, gauss2, dog, size);
    
    // ASCII edge detection using DoG (matching GPU version)
    asciiEdgeCPU(dog, equalized, ascii, width, height, 100, dsWidth, dsHeight);

    FILE* out = fopen(argv[2], "w");
    if (!out) {
        printf("Error opening output file.\n");
        return 1;
    }

    for (int y = 0; y < dsHeight; y++) {
        for (int x = 0; x < dsWidth; x++) {
            char ch = ascii[y * dsWidth + x];
            fputc(ch, out);
            fputc(ch, out);  // Double character width
        }
        fputc('\n', out);
    }
    fclose(out);

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Saved to %s\n", argv[2]);
    printf("Execution time: %.3f seconds\n", elapsed);

    stbi_image_free(img);
    free(equalized);
    free(gauss1);
    free(gauss2);
    free(dog);
    free(ascii);

    return 0;
}
