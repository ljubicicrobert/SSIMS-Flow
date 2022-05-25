#pragma once

#define EXTERN_C extern "C"
#define DLL_API EXTERN_C __declspec(dllexport)

typedef unsigned char Byte;
