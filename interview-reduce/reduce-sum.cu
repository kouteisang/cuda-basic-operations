#include "../header.h"

using ll = unsigned long long;

__device__ __forceinline__
ll warp_reduce_sum(ll x){
    int lane_id = threadIdx.x & 31;
    for(int offset = 16; offset > 0; offset >>= 1){
        x += __shfl_down_sync(0xFFFFFFFFu, x, offset);
    }
    return x;
}

__global__ void reduce_block_atomic(int* __restrict__ d_in, ll* d_out, int n){

    __shared__ ll sum_warp[256/32];
    int num_of_warp = 256/32;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;
    ll local = 0;

    for(int t = tid; t < n; t += gridDim.x * blockDim.x){
        local += (ll)d_in[t];
    }

    ll n_local = warp_reduce_sum(local);
    if(lane_id == 0) {
        sum_warp[warp_id] =  n_local;
    }
    __syncthreads();

    if(warp_id == 0){
        ll x = lane_id < num_of_warp ? sum_warp[lane_id] : 0;
        ll sum_in_warp = warp_reduce_sum(x);
        if(lane_id == 0) atomicAdd(d_out, sum_in_warp); 
    }
}

int main() {
    int N = 1<<20;
    std::vector<int> h(N, 1);
    ll h_sum;
    int* d_in = nullptr;
    ll* d_out = nullptr;

    // TODO: cudaMalloc, cudaMemcpy
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMemcpy(d_in, h.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, 1 * sizeof(ll));
    // TODO: cudaMemset d_out = 0
    cudaMemset(d_out, 0, sizeof(ll));

    int BLOCK = 256;
    int GRID  = (N + BLOCK - 1) / BLOCK;
    GRID = min(GRID, 4096);
    

    reduce_block_atomic<<<GRID, BLOCK>>>(d_in, d_out, N);
    cudaError_t e = cudaGetLastError();
    if(e != cudaSuccess){
        printf("Lunch Error %s\n", cudaGetErrorString(e));
    }
    
    // TODO: check launch error + cudaDeviceSynchronize
    cudaDeviceSynchronize();
    // TODO: copy back and print (expected sum = N)
    cudaMemcpy(&h_sum, d_out, sizeof(ll), cudaMemcpyDeviceToHost);
    printf("sum = %llu", h_sum);
    // TODO: free
    cudaFree(d_in);
    cudaFree(d_out);
}
