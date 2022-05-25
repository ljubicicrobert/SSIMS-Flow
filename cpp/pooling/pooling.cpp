#include "pooling.h"

float mag_pool(float* array, int size, float k) {
    int numValid = 0;
    float sum = 0;
    float maskedSum = 0;
    Byte* mask = new Byte[size];

    for (int i = 0; i < size; i++) {
        sum += array[i];
    }

    float mean = sum / size;

    for (int i = 0; i < size; i++) {
        if (array[i] >= k * mean) {
            numValid++;
            maskedSum += array[i];
        }
    }

    return maskedSum / numValid;
}
