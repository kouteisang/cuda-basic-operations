#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace std;

#define tile_size 16

__global__ void matmul_tile(int* d_a, int* d_b, int* d_out, int m, int n, int k){

    __shared__ int sh_a[tile_size][tile_size];
    __shared__ int sh_b[tile_size][tile_size];

    int c_row = tile_size * blockIdx.y + threadIdx.y;
    int c_col = tile_size * blockIdx.x + threadIdx.x;
    int sum = 0;

    for(int t = 0; t < (n+tile_size-1)/tile_size; t ++){

        int a_col = t * tile_size + threadIdx.x;
        int b_row = t * tile_size + threadIdx.y;

        sh_a[threadIdx.y][threadIdx.x] = (c_row < m && a_col < n) ? d_a[c_row * n + a_col] : 0;
        sh_b[threadIdx.y][threadIdx.x] = (b_row < n && c_col < k) ? d_b[b_row * k + c_col] : 0;
        
        __syncthreads();
        
        for(int i = 0; i < tile_size; i ++){
            sum += sh_a[threadIdx.y][i] * sh_b[i][threadIdx.x];
        }
        __syncthreads();

    }

    if(c_row < m && c_col < k) d_out[c_row * k + c_col] = sum;
}


int main(int argc, char** argv){
  
    int m, n, k;

    for(int i = 1; i < argc; i ++){
        string arg = argv[i];
        if(arg == "-m"){
            m = stoi(argv[++i]);
        }else if(arg == "-n"){
            n = stoi(argv[++i]);
        }else if(arg == "-k"){
            k = stoi(argv[++i]);
        }
    }

    vector<int> a(m * n);
    vector<int> b(n * k);
    vector<int> h_out(m * k);

    for(int i = 0; i < m * n; i ++) a[i] = 1;
    for(int i = 0; i < n * k; i ++) b[i] = 1;
    
    int *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, (m*n) * sizeof(int));
    cudaMalloc(&d_b, (n*k) * sizeof(int));
    cudaMalloc(&d_out, (m*k) * sizeof(int));

    
    cudaMemcpy(d_a, a.data(), (m*n) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), (n*k) * sizeof(int), cudaMemcpyHostToDevice);
    
    
    dim3 blk_num((k+tile_size-1)/tile_size, (m+tile_size-1)/tile_size);
    dim3 blk_dim(tile_size, tile_size);
    matmul_tile<<<blk_num, blk_dim>>>(d_a, d_b, d_out,  m, n, k);

    
    cudaMemcpy(h_out.data(), d_out, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < m; i ++){
        for(int j = 0; j < k; j ++){
            printf("%d ", h_out[i*k + j]);
        }
        printf("\n");
    }   
}