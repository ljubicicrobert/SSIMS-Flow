#include "pooling.h"

double mag_pool(float* array, unsigned int size, double k) {
    unsigned int numValid = 0;
    double sum = 0;
    double maskedSum = 0;
    byte* mask = new byte[size];

    for (int i = 0; i < size; i++) {
        sum += array[i];
    }

    double mean = sum / size;

    for (int i = 0; i < size; i++) {
        if (array[i] >= k * mean) {
            numValid++;
            maskedSum += array[i];
        }
    }

    return maskedSum / numValid;
}
