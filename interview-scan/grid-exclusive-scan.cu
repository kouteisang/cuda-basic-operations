#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cub/cub.cuh>

using namespace std;

#define BLK_NUM 3
#define BLK_DIM 128

__device__ __forceinline__ 
int warp_exclusive_scan(int x){
    int lane_id = threadIdx.x & 31;
    int o = x;
    for(int offset = 1; offset < 32; offset <<= 1){
        int y = __shfl_up_sync(0xFFFFFFFFu, x, offset);    
        if(lane_id >= offset) x += y;    
    }
    return x-o;

}

__global__ void block_scan(int* d_in, int* d_out, int N, int* block_sum){
    
    __shared__ int warp_total[BLK_DIM / 32];
    __shared__ int warp_offset[BLK_DIM / 32];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;

    int x = tid < N ? d_in[tid] : 0;
    int ex = warp_exclusive_scan(x);

    int warp_sum = ex + x;
    if(lane_id == 31) warp_total[warp_id] = warp_sum;
    __syncthreads();

    if(warp_id == 0){
        int w = lane_id < (BLK_DIM / 32) ? warp_total[lane_id] : 0;
        int ew = warp_exclusive_scan(w);
        if(lane_id < BLK_DIM / 32) warp_offset[lane_id] = ew;

        if(lane_id == 0){
            block_sum[blockIdx.x] = warp_offset[BLK_DIM / 32 - 1] + warp_total[BLK_DIM / 32 - 1];
        }
    }
    __syncthreads();

    if(tid < N) d_out[tid] = ex + warp_offset[warp_id];
}

__global__ void block_add(int* d_out, int* block_offset, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int t = tid; t < N; t += blockDim.x * gridDim.x){
        d_out[t] += block_offset[blockIdx.x];
    }
}

int main(){
    int N = 300;
    vector<int> h_in(N), h_out(N);

    for(int i = 0; i < N; i ++){
        h_in[i] = 1;
    }

    int *d_in, *d_out;
    int *block_sum, *block_offset;
    cudaMalloc(&block_sum, BLK_NUM * sizeof(int));  
    cudaMemset(&block_sum, 0, sizeof(int) * BLK_NUM);
    
    cudaMalloc(&block_offset, BLK_NUM * sizeof(int));  
    cudaMemset(&block_offset, 0, sizeof(int) * BLK_NUM);
    
    
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    
    block_scan<<<BLK_NUM,BLK_DIM>>>(d_in, d_out, N, block_sum); 

    void* d_temp = nullptr;;
    size_t temp_bytes = 0;

    // query
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, block_sum, block_offset, BLK_NUM);
    cudaMalloc(&d_temp, temp_bytes);
    // run
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, block_sum, block_offset, BLK_NUM);

    block_add<<<BLK_NUM, BLK_DIM>>>(d_out, block_offset, N); 
    
    cudaMemcpy(h_out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 每个 warp 的输出应该是 0..31
    for (int i = 0; i < N; i++) {
        printf("%d%s", h_out[i], (i % 32 == 31) ? "\n" : " ");
    }

    cudaFree(d_in);
    cudaFree(d_out);
}