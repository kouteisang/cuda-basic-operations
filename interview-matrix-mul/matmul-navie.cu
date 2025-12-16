#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace std;



__global__ void matmul(const int* __restrict__ d_a, const int* __restrict__ d_b, int* __restrict__ d_out, int m, int n, int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < k){
        int sum = 0;
        for(int i = 0; i < n; i ++){
            sum += d_a[row * n + i] * d_b[i * k + col];
        }
        d_out[row*k+col] =  sum;
    }
}

__global__ void matmul2(const int* __restrict__ d_a, const int* __restrict__ d_b, int* __restrict__ d_out, int m, int n, int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for(int r = row; r < m; r += gridDim.y * blockDim.y){

        for(int c = col; c < k; c += gridDim.x * blockDim.x){

            int sum = 0;
            for(int i = 0; i < n; i ++){
                sum += d_a[r * n + i] * d_b[i * k + c];
            }
            d_out[r*k+c] =  sum;
            
        }
    }
        

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
    
    // dim3 blk_num((k + 8 - 1) / 8, (m + 8 - 1) / 8);
    // dim3 blk_dim(8, 8);
    // matmul<<<blk_num, blk_dim>>>(d_a, d_b, d_out,  m, n, k);

    dim3 blk_num2(1, 1);
    dim3 blk_dim2(2, 2);
    matmul2<<<blk_num2, blk_dim2>>>(d_a, d_b, d_out,  m, n, k);

    
    cudaMemcpy(h_out.data(), d_out, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < m; i ++){
        for(int j = 0; j < k; j ++){
            printf("%d ", h_out[i*k + j]);
        }
        printf("\n");
    }
}