#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000 // Number of parabolas

#define MIN_COEFF -10.0F // Inclusive
#define MAX_COEFF +10.0F // Exclusive

void fillWithRandomFloats(float *arr, int size, float min_val, float max_val)
{
    for (int i = 0; i < size; i++) 
    {
        float rand_float = ((float) rand()) / ((float) RAND_MAX);
    }
}

int main()
{
    srand(time(NULL));

    int size = 3*N;
    float *coeffs = (float*) calloc(size, sizeof(float));

    fillWithRandomFloats(coeffs, N, MIN_COEFF, MAX_COEFF);


    printf("Hello World!\n");
    return 0;
}