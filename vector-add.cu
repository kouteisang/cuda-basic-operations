#include "header.h"

#define BLK_NUM 2
#define BLK_DIM 128

__global__ void vector_add(int* __restrict__ d_a, const int* __restrict__ d_b, int n){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int t = tid; t < n; t += blockDim.x * gridDim.x){
        d_a[t] = d_a[t] + d_b[t];
    }
}

int main(){

    int N = 300;
    vector<int> h_a(N), h_b(N);
    int *d_a, *d_b;

    for(int i = 0; i < N; i ++) {
        h_a[i] = i;
        h_b[i] = i;
    }
    
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));

    cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    vector_add<<<BLK_NUM, BLK_DIM>>>(d_a, d_b, N);

    cudaMemcpy(h_a.data(), d_a, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaError_t e = cudaGetLastError();
    if(e != cudaSuccess){
        printf("Error!!!, %s", cudaGetErrorString(e));
    }

    for(int i = 0; i < N; i ++){
        printf("i = %d ", h_a[i]);
        if(i % 10 == 0) printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
}