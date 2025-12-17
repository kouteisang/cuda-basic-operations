#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace std;

__global__ void hist_construct_share(const int* __restrict__ d_in, int* __restrict__ hist, int n){

    extern __shared__ int sh[];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
   

    for(int t = threadIdx.x; t < n; t += blockDim.x) sh[t] = 0;
    __syncthreads();


    for(int t = tid; t < n; t += step) {
        atomicAdd(&sh[d_in[t]], 1);
    }
    __syncthreads();

    for(int t = threadIdx.x; t < n; t += blockDim.x){
        atomicAdd(&hist[t], sh[t]);
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
    hist_construct_share<<<grid, 256, (size_t)n * sizeof(int)>>>(d_in, hist, n);

    cudaMemcpy(out_hist.data(), hist, n * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i ++){
        printf("number %d, frequency %d \n",i, out_hist[i]);
    }
}