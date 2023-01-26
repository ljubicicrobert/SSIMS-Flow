#pragma once

#define EXTERN_C extern "C"
#define DLL_API EXTERN_C __declspec(dllexport)

DLL_API double mag_pool(float* array, size_t size, double m, int iter);
