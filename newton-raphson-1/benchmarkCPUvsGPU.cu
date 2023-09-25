#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <fstream>

#define MAX_N 10000000 // Max number of parabolas
#define N_INC 10000
#define K 3 // Number of coefficients

#define MIN_COEFF -10.0F // Inclusive
#define MAX_COEFF +10.0F // Exclusive

#define MAX_ITERATIONS 100
#define TOLERANCE 1E-6
#define EPSILON 1E-6

#define TPB 32

void fillWithRandomFloats(float * arr, const int size, const float min_val, const float max_val)
{
    for (int i = 0; i < size; i++) 
    {
        float rand_float = (max_val - min_val)*(((float) rand()) / ((float) RAND_MAX)) + min_val;
        arr[i] = rand_float;
    }
}

__host__ __device__
inline int coeff_index(const int n_parabola, const int n_coeff)
{
    return K*n_parabola + n_coeff;
}

__host__ __device__
inline float parabola(const float x, const float * coeffs, const float n)
{
    const int cindex = coeff_index(n, 0);
    const float A = coeffs[cindex];
    const float B = coeffs[cindex + 1];
    const float C = coeffs[cindex + 2];
    return A*x*x + B*x + C;
}

__host__ __device__
inline float d_parabola(const float x, const float * coeffs, const float n)
{
    const int cindex = coeff_index(n, 0);
    const float A = coeffs[cindex];
    const float B = coeffs[cindex + 1];
    return 2.0F*A*x + B;
}

__host__ __device__
inline float d2_parabola(const float x, const float * coeffs, const int n)
{
    const float A = coeffs[coeff_index(n, 0)];
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

void newton_raphson_parabola_CPU(const float * coeffs, const float * x0s, float * roots, const float size)
{
    for (int p = 0; p < size; p++)
    {
        roots[p] = get_root(coeffs, p, x0s[p]);
    }
}

__global__
void newton_raphson_parabola_GPU(const float * coeffs, const float * x0s, float * roots, const float size)
{
    const int p = blockIdx.x*blockDim.x + threadIdx.x;
    roots[p] = get_root(coeffs, p, x0s[p]);
}

void print_n_roots(const float * coeffs, const float *roots, const float n)
{
    for (int p = 0; p < n; p++)
    {
        const float A = coeffs[coeff_index(p, 1)];
        const float B = coeffs[coeff_index(p, 2)];
        const float C = coeffs[coeff_index(p, 3)];
        float root = roots[p];
        printf("%.3fx^2 + %.3fx + %.3f has a root of %.3f\n", A, B, C, root);
    }
}

int main()
{
    srand(time(NULL));

    std::ofstream myfile;
    myfile.open("/app/CPUvsGPU.csv");
    myfile.clear();

    for (int N = 0; N < MAX_N; N += N_INC)
    {

        printf("%d\n", N);
        

        myfile << N;
        myfile << ",";

        // CPU

        float * x0s = (float*) calloc(N, sizeof(float));
        float * coeffs = (float*) calloc(K*N, sizeof(float));
        float * roots = (float*) calloc(N, sizeof(float));

        fillWithRandomFloats(x0s, N, MIN_COEFF, MAX_COEFF);
        fillWithRandomFloats(coeffs, K*N, MIN_COEFF, MAX_COEFF);

        auto time_start = std::chrono::high_resolution_clock::now();

        newton_raphson_parabola_CPU(coeffs, x0s, roots, N);

        auto time_end = std::chrono::high_resolution_clock::now();
        auto delta_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start);

        myfile << delta_time.count();
        myfile << ",";

        //printf("CPU Time: %lu ns\n", delta_time.count());
        //print_n_roots(coeffs, roots, 3);

        // GPU

        float * dev_x0s = 0;
        float * dev_coeffs = 0;
        float * dev_roots = 0;

        cudaMallocManaged(&dev_x0s, N*sizeof(float));
        cudaMallocManaged(&dev_coeffs, K*N*sizeof(float));
        cudaMallocManaged(&dev_roots, N*sizeof(float));

        time_start = std::chrono::high_resolution_clock::now();

        cudaMemcpy(dev_x0s, x0s, N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_coeffs, coeffs, K*N*sizeof(float), cudaMemcpyHostToDevice);

        newton_raphson_parabola_GPU<<<N/TPB, TPB>>>(dev_coeffs, dev_x0s, dev_roots, N);
        cudaDeviceSynchronize();

        cudaMemcpy(roots, dev_roots, N, cudaMemcpyDeviceToHost);

        time_end = std::chrono::high_resolution_clock::now();
        delta_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start);

        myfile << delta_time.count();

        //printf("GPU Time: %lu ns\n", delta_time.count());
        //print_n_roots(dev_coeffs, dev_roots, 3);

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