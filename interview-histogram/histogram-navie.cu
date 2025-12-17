#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace std;

__global__ void hist_construct(const int* __restrict__ d_in, int* __restrict__ hist, const int n){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int t = tid; t < n; t += blockDim.x * gridDim.x){
        atomicAdd(&hist[d_in[t]], 1);
    }
}

int main(int argc, char** argv){
    int n = 0;

    for(int i = 1; i < argc; i ++){
        string arg = argv[i];
        if(arg == "-n"){
            n = stoi(argv[++i]);
        }
    }

    vector<int> in(n);
    int* hist, *d_in;
    vector<int> out_hist(n);


    for(int i = 0; i < n; i ++){
        in[i] = rand() % n;
    }
    
    cudaMalloc(&hist, n * sizeof(int));
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMemcpy(d_in, in.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(hist, 0, sizeof(int) * n);

    int grid = (n+256-1) / 256;
    hist_construct<<<grid, 256>>>(d_in, hist, n);

    cudaMemcpy(out_hist.data(), hist, n * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i ++){
        printf("number %d, frequency %d \n",i, out_hist[i]);
    }
}