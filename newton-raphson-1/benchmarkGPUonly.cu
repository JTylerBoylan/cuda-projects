#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <assert.h>

#define MAX_N 100000000 // Max number of parabolas
#define N_INC 100000
#define K 3 // Number of coefficients

#define MIN_COEFF -10.0F // Inclusive
#define MAX_COEFF +10.0F // Exclusive

#define MAX_ITERATIONS 100
#define TOLERANCE 1E-6
#define EPSILON 1E-6

#define TPB 256

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    size_t free_, total;
    cudaMemGetInfo(&free_, &total);
    printf("Free: %zu, Total: %zu\n", free_, total);
    assert(result == cudaSuccess);
  }
  return result;
}

void fillWithRandomFloats(float * arr, const int size, const float min_val, const float max_val)
{
    for (int i = 0; i < size; i++) 
    {
        float rand_float = (max_val - min_val)*(((float) rand()) / ((float) RAND_MAX)) + min_val;
        arr[i] = rand_float;
    }
}

__host__ __device__
inline int coeff_index(const int n_parabola)
{
    return K*n_parabola;
}

__host__ __device__
inline float parabola(const float x, const float * coeffs, const int n)
{
    const int cindex = coeff_index(n);
    const float A = coeffs[cindex];
    const float B = coeffs[cindex + 1];
    const float C = coeffs[cindex + 2];
    return A*x*x + B*x + C;
}

__host__ __device__
inline float d_parabola(const float x, const float * coeffs, const int n)
{
    const int cindex = coeff_index(n);
    const float A = coeffs[cindex];
    const float B = coeffs[cindex + 1];
    return 2.0F*A*x + B;
}

__host__ __device__
inline float d2_parabola(const float x, const float * coeffs, const int n)
{
    const float A = coeffs[coeff_index(n)];
    return 2.0F*A;
}

__host__ __device__
float get_root(const float * coeffs, const int n, float x0)
{
    for (int i = 0; i < MAX_ITERATIONS; i++)
    {

        const float y = d_parabola(x0, coeffs, n);
        const float y_prime = d2_parabola(x0, coeffs, n);

        if (abs(y_prime) < EPSILON)
        {
            break;
        }

        const float x1 = x0 - (y / y_prime);

        if (abs(x1 - x0) <= TOLERANCE)
        {
            return x1;
        }

        x0 = x1;
    }

    return INFINITY;
}

void newton_raphson_parabola_CPU(const float * coeffs, const float * x0s, float * roots, const int size)
{
    for (int p = 0; p < size; p++)
    {
        roots[p] = get_root(coeffs, p, x0s[p]);
    }
}

__global__
void newton_raphson_parabola_GPU(const float * coeffs, const float * x0s, float * roots, const int size)
{
    const int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p < size)
        roots[p] = get_root(coeffs, p, x0s[p]);
}

void print_n_roots(const float * coeffs, const float *roots, const int n)
{
    for (int p = 0; p < n; p++)
    {
        const int cindex = coeff_index(n);
        const float A = coeffs[cindex];
        const float B = coeffs[cindex + 1];
        const float C = coeffs[cindex + 2];
        float root = roots[p];
        printf("%.3fx^2 + %.3fx + %.3f has a root of %.3f\n", A, B, C, root);
    }
}

int main()
{
    srand(time(NULL));

    std::ofstream myfile;
    myfile.open("/app/GPUonly.csv");
    myfile.clear();

    for (int N = N_INC; N <= MAX_N; N += N_INC)
    {

        printf("%d\n", N);
        

        myfile << N;
        myfile << ",";

        float * x0s = (float*) calloc(N, sizeof(float));
        float * coeffs = (float*) calloc(K*N, sizeof(float));
        float * roots = (float*) calloc(N, sizeof(float));

        fillWithRandomFloats(x0s, N, MIN_COEFF, MAX_COEFF);
        fillWithRandomFloats(coeffs, K*N, MIN_COEFF, MAX_COEFF);

        float * dev_x0s = 0;
        float * dev_coeffs = 0;
        float * dev_roots = 0;

        checkCuda( cudaMallocManaged(&dev_x0s, N*sizeof(float)) );
        checkCuda( cudaMallocManaged(&dev_coeffs, K*N*sizeof(float)) );
        checkCuda( cudaMallocManaged(&dev_roots, N*sizeof(float)) );

        size_t number_of_blocks = (N + TPB - 1) / TPB;

        auto time_start = std::chrono::high_resolution_clock::now();

        checkCuda( cudaMemcpy(dev_x0s, x0s, N*sizeof(float), cudaMemcpyHostToDevice) );
        checkCuda( cudaMemcpy(dev_coeffs, coeffs, K*N*sizeof(float), cudaMemcpyHostToDevice) );

        newton_raphson_parabola_GPU<<<number_of_blocks, TPB>>>(dev_coeffs, dev_x0s, dev_roots, N);
        checkCuda(cudaGetLastError());

        checkCuda( cudaDeviceSynchronize() );

        checkCuda( cudaMemcpy(roots, dev_roots, N*sizeof(float), cudaMemcpyDeviceToHost) );

        auto time_end = std::chrono::high_resolution_clock::now();
        auto delta_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start);

        myfile << delta_time.count();

        //printf("GPU Time: %lu ns\n", delta_time.count());
        //print_n_roots(coeffs, roots, 1);

        myfile << "\n";

        // FREE

        free(x0s);
        free(roots);
        free(coeffs);

        cudaFree(dev_x0s);
        cudaFree(dev_roots);
        cudaFree(dev_coeffs);
    }

    return 0;
}