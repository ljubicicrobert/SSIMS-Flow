#define EXTERN_C extern "C"
#define DLL_API EXTERN_C __declspec(dllexport)

typedef unsigned char Byte;

DLL_API float mag_pool(float* array, int size, float k);
