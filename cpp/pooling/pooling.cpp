#include "pooling.h"

double mag_pool(float* array, size_t size, double m, int iter) {
    size_t numValid;
    double sum = 0;
    double maskedSum;
    double mean;

    if (m < 0) {
        for (size_t i = 0; i < size; i++) {
            sum += array[i];
        }

        mean = sum / size;
    }
    else {
        mean = m;
    }

    for (size_t i = 0; i < iter; i++) {
        numValid = 0;
        maskedSum = 0;

        for (size_t j = 0; j < size; j++) {
            if (array[j] >= mean) {
                numValid++;
                maskedSum += array[j];
            }
        }

        mean = maskedSum / numValid;
    }

    return mean;
}
