#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace std;

__global__ void hist_construct_stage1(int* __restrict__ global_hist, const int* __restrict__  d_in, int n, int b){

    int* l_host = global_hist + blockIdx.x * b;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;

    for(int t = tid; t < n; t += step){
        atomicAdd(&l_host[d_in[t]], 1);
    }
}

__global__ void hist_construct_stage2(int* __restrict__ global_hist, int* __restrict__  hist, int n, int b){

    int* l_host = global_hist + blockIdx.x * b;

    int bin_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(bin_id >= b) return ;
    int sum = 0;

    for(int blockId = 0; blockId < gridDim.x; blockId ++){
        sum += global_hist[blockId * b + bin_id];
    }
    hist[bin_id] = sum;
}




int main(int argc, char** argv){

    int n = 0;
    int b = 0;

    for(int i = 1; i < argc; i ++){
        string arg = argv[i];
        if(arg == "-n"){
            n = stoi(argv[++i]);
        }else if(arg == "-b"){
            b = stoi(argv[++i]);
        }
    }

    vector<int> in(n), out_hist(b);
    int* hist, *d_in;


    for(int i = 0; i < n; i ++){
        in[i] = rand() % b;
    }
    
    cudaMalloc(&hist, b * sizeof(int));
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMemcpy(d_in, in.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(hist, 0, sizeof(int) * b);

    int* global_hist;

    int grid = (n+256-1) / 256;

    cudaMalloc(&global_hist, b * grid * sizeof(int));
    cudaMemset(global_hist, 0, b * grid * sizeof(int));
    
    hist_construct_stage1<<<grid, 256>>>(global_hist, d_in, n, b);
    hist_construct_stage2<<<grid, 256>>>(global_hist, hist, n, b);
    

    cudaMemcpy(out_hist.data(), hist, b * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < b; i ++){
        printf("number %d, frequency %d \n",i, out_hist[i]);
    }
}