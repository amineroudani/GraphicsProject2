#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Random engine and normal distribution for direction vectors
static std::default_random_engine rngEngine(10);
static std::normal_distribution<double> normalDist(0, 1);

/**
 * @brief Clamps a value between a minimum and maximum value.
 * 
 * @param value The value to clamp.
 * @param minValue The minimum value.
 * @param maxValue The maximum value.
 * @return The clamped value.
 */
unsigned char clampValue(double value, double minValue, double maxValue) {
    if (value < minValue) return static_cast<unsigned char>(minValue);
    if (value > maxValue) return static_cast<unsigned char>(maxValue);
    return static_cast<unsigned char>(value);
}

/**
 * @brief Performs sliced matching between source and target images.
 * 
 * @param srcImg The source image.
 * @param tgtImg The target image.
 * @param width The width of the images.
 * @param height The height of the images.
 * @param resultImg The result image.
 */
void performSlicedMatching(const double* srcImg, double* tgtImg, int width, int height, double* resultImg) {
    memcpy(resultImg, srcImg, width * height * 3 * sizeof(double));

    for (int iter = 0; iter < 100; ++iter) {
        double xDir = normalDist(rngEngine);
        double yDir = normalDist(rngEngine);
        double zDir = normalDist(rngEngine);
        double normFactor = std::sqrt(xDir * xDir + yDir * yDir + zDir * zDir);
        xDir /= normFactor;
        yDir /= normFactor;
        zDir /= normFactor;

        std::vector<std::pair<double, int>> projectedSrc(width * height);
        std::vector<double> projectedTgt(width * height);

        for (int i = 0; i < width * height; ++i) {
            double srcProjection = resultImg[i * 3] * xDir + resultImg[i * 3 + 1] * yDir + resultImg[i * 3 + 2] * zDir;
            double tgtProjection = tgtImg[i * 3] * xDir + tgtImg[i * 3 + 1] * yDir + tgtImg[i * 3 + 2] * zDir;
            projectedSrc[i] = std::make_pair(srcProjection, i);
            projectedTgt[i] = tgtProjection;
        }

        std::sort(projectedSrc.begin(), projectedSrc.end());
        std::sort(projectedTgt.begin(), projectedTgt.end());

        for (int i = 0; i < width * height; ++i) {
            double shiftAmount = projectedTgt[i] - projectedSrc[i].first;
            int idx = projectedSrc[i].second;
            resultImg[idx * 3] += shiftAmount * xDir;
            resultImg[idx * 3 + 1] += shiftAmount * yDir;
            resultImg[idx * 3 + 2] += shiftAmount * zDir;
        }
    }
}

int main() {
    int width, height, channels;

    // Load source and target images
    unsigned char* sourceImgData = stbi_load("8733654151_b9422bb2ec_k.jpg", &width, &height, &channels, STBI_rgb);
    unsigned char* targetImgData = stbi_load("redim.jpg", &width, &height, &channels, STBI_rgb);

    std::vector<double> sourceImg(width * height * 3);
    std::vector<double> targetImg(width * height * 3);

    for (int i = 0; i < width * height * 3; ++i) {
        sourceImg[i] = static_cast<double>(sourceImgData[i]);
        targetImg[i] = static_cast<double>(targetImgData[i]);
    }

    std::vector<double> outputImg(width * height * 3);
    performSlicedMatching(sourceImg.data(), targetImg.data(), width, height, outputImg.data());

    std::vector<unsigned char> finalImg(width * height * 3, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            finalImg[(y * width + x) * 3] = clampValue(outputImg[(y * width + x) * 3], 0, 255);
            finalImg[(y * width + x) * 3 + 1] = clampValue(outputImg[(y * width + x) * 3 + 1], 0, 255);
            finalImg[(y * width + x) * 3 + 2] = clampValue(outputImg[(y * width + x) * 3 + 2], 0, 255);
        }
    }

    stbi_write_png("output_image.png", width, height, 3, finalImg.data(), 0);

    stbi_image_free(sourceImgData);
    stbi_image_free(targetImgData);

    return 0;
}
