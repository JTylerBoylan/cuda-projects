#ifndef KKT_SOLVER_GENERATED_LOOKUP_CU_
#define KKT_SOLVER_GENERATED_LOOKUP_CU_

#define NUM_OBJECTIVES 2
#define NUM_VARIABLES 8

typedef float (*intercept_ptr)(float*);

__device__ float COST(float * w)
{
    // ...
}

__constant__ float w0_0_0;
__device__ float j_0_0(float * w)
{
    // ...
}

__constant__ float w0_0_1;
__device__ float j_0_1(float * w)
{
    // ...
}

// ...

__constant__ float w0_N_M;
__device__ float j_N_M(float * w)
{
    // ...
}

__constant__ intercept_ptr LOOKUP_INTERCEPT[NUM_OBJECTIVES*NUM_VARIABLES]; // to be initialized from host
__constant__ float LOOKUP_INITIAL[NUM_OBJECTIVES*NUM_VARIABLES];  // to be initialized from host

__global__ void setup_lookup(intercept_ptr* lookup_intercept, float* lookup_initial)
{
    lookup_intercept[0] = j_0_0;
    lookup_intercept[1] = j_0_1;
    // ...
    lookup_initial[0] = w0_0_0;
    lookup_initial[1] = w0_0_1;
    // ...
}

#endif