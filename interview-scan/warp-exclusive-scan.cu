#include <iostream>
#include <cuda_runtime.h>
#include <vector>
using namespace std;

// This function is about the exclusive scan in a warp of thread

// 一个F代码4个active thread
// 比如0x0000000F 代表lane_id 0,1,2,3 active，其他都不active
// 比如0x000000F0 代表lane_id 4,5,6,7 active，其他都不active
// <<= 左移，相当于*2 
// =>> 右移，相当于/2
// __shfl_up_sync(mask, x, offset) = 拿到当前位置-offset的pos的数值
__global__ void warp_scan(int* d_in, int* d_out, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int x = tid < N ? d_in[tid] : 0;
    int origin  = x;
    for(int offset = 1; offset < 32; offset <<= 1){
        int y = __shfl_up_sync(0xFFFFFFFFu, x, offset);
        if(lane_id - offset >= 0) x+= y;
    }

    if(tid < N) d_out[tid] = x - origin;
    
}

int main(){
    const int N = 128;
    
    std::vector<int> h_in(N);

    for(int i = 0; i < N; i ++){
        h_in[i] = 1;
    }

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    warp_scan<<<1, 128>>>(d_in, d_out, N);

    cudaMemcpy(h_in.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i ++){
        printf("%d, ", h_in[i]);
    }
}