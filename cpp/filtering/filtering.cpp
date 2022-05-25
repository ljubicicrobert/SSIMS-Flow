#include "filtering.h"
#include <math.h>
#include <vector>
#include <iostream>

template<class T>
void print(T t) {
    std::cout << +t << std::endl;
}

template<class S, class T>
void print(S s, T t) {
    std::cout << +s << +t << std::endl;
}

void intensity_capping(byte* array, unsigned int array_size, double n_std) {
    std::vector<unsigned int> bytes_map(256, 0);
    byte pixel = 0;
    byte pixel_min = 255;

    unsigned long long int sum = 0;
    unsigned int pixel_median = 0;
    unsigned int i;
    
    double pixel_as_double = 0;
    double pixel_min_as_double;

    // Find mean and prepare for median search
    for (i = 0; i < array_size; i++) {
        pixel = array[i];
        bytes_map[pixel] += 1;          // For median search
        sum += pixel;                   // For mean value

        if (pixel < pixel_min) {        // For normalization
            pixel_min = pixel;
        }
    }

    pixel_min_as_double = pixel_min;
    byte pixel_mean = sum / array_size + 0.5;

    // Finding median through pixel value histogram
    sum = 0;
    while (sum <= array_size / 2) {
        sum += bytes_map[pixel_median++];
    }

    pixel_median--;

    // For standard deviation
    sum = 0;
    for (i = 0; i < array_size; i++) {
        sum += pow(array[i] - pixel_mean, 2);
    }

    double stdev = sqrt(sum / array_size);
    double cap = pixel_median - n_std * stdev + 0.5;
    byte cap_byte = cap;

    // Perform pixel intensity capping
    for (i = 0; i < array_size; i++) {
        if (array[i] > cap_byte) {
            array[i] = cap_byte;
        }

        pixel_as_double = (1.0 - (cap - array[i]) / (cap - pixel_min_as_double)) * 255.0;
        array[i] = pixel_as_double;
    }
}
