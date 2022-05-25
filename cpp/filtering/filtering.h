#pragma once

#define EXTERN_C extern "C"
#define DLL_API EXTERN_C __declspec(dllexport)

typedef unsigned char byte;

DLL_API void intensity_capping(byte* array, unsigned int array_size, double n_std);
