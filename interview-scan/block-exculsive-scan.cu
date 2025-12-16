#include <iostream>
#include <cuda_runtime.h>
#include <vector>
using namespace std;

#define BLK_DIM 128

__device__ __forceinline__ 
int warp_scan(int x){
    int lane_id = threadIdx.x & 31;
    int o = x;
    for(int offset = 1; offset < 32; offset <<= 1){
        int y = __shfl_up_sync(0xFFFFFFFFu, x, offset);
        if(lane_id >= offset) x += y;
    }
    return x - o;
}


__global__ void test_block_scan(int* d_in, int* d_out, int N){
    __shared__ int warp_sum[BLK_DIM / 32]; // blockdim / warp_size
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int x = tid < N ? d_in[tid] : 0;
    int origin = x;

    int ex = warp_scan(x);

    int warp_total = ex + x;
    if(lane_id == 31) warp_sum[warp_id] = warp_total;
    __syncthreads();

    if(warp_id == 0){
        int w = (lane_id < BLK_DIM/32) ? warp_sum[lane_id] : 0;
        int w_ex = warp_scan(w);
        if(lane_id < BLK_DIM/32) warp_sum[lane_id] = w_ex; 
    }
    __syncthreads();

    if(tid < N) d_out[tid] = ex + warp_sum[warp_id];

}

int main(){

    const int N = 128; // 4 warps
    std::vector<int> h_in(N), h_out(N);
    for (int i = 0; i < N; i++) h_in[i] = 1; // easy check

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    test_block_scan<<<1, BLK_DIM>>>(d_in, d_out, N);
    cudaMemcpy(h_out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 每个 warp 的输出应该是 0..31
    for (int i = 0; i < N; i++) {
        printf("%d%s", h_out[i], (i % 32 == 31) ? "\n" : " ");
    }

    cudaFree(d_in); cudaFree(d_out);

}