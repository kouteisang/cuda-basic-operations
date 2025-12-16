#include "header.h"

using ll = unsigned long long;
#define BLK_DIM 128

__device__ __forceinline__
ll reduce_sum_in_warp(ll local_sum){
    for(int offset = 16; offset > 0; offset >>= 1){
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, offset);
    }
    return local_sum;
}

template<typename T>
__global__ void reduce_sum(T* d_in, ll* d_tmp, int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;
    int step = gridDim.x * blockDim.x;
    ll local_sum = 0;
    __shared__ ll sh_sum[BLK_DIM / 32];

    for(int t = tid; t < N; t += step){
        local_sum += (ll)d_in[t];
    }

    local_sum = reduce_sum_in_warp(local_sum);

    if(lane_id == 0){
        sh_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if(warp_id == 0){
        ll x = lane_id < (BLK_DIM/32) ? sh_sum[lane_id] : 0;
        x = reduce_sum_in_warp(x);
        if(lane_id == 0) d_tmp[blockIdx.x] = x;

    }
}


int main(){
    int N = 1<<20;
    std::vector<int> h(N, 1);

    for(int i = 0; i < N; i ++){
        h[i] = 1;
    }

    int *d_in;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMemcpy(d_in, h.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    int blk_num = (N + BLK_DIM - 1) / BLK_DIM;
    ll *d_tmp;
    cudaMalloc(&d_tmp, sizeof(ll) * blk_num);
    reduce_sum<int><<<blk_num, BLK_DIM>>>(d_in, d_tmp, N);

    int len = blk_num;
    ll* d_out;
    while(len > 1){
        blk_num = (len + BLK_DIM - 1) / BLK_DIM;
        cudaMalloc(&d_out, blk_num * sizeof(ll));
        reduce_sum<ll><<<blk_num, BLK_DIM>>>(d_tmp, d_out, len);
        d_tmp = d_out;
        d_out = nullptr;
        len = blk_num;
    }

    ll sum;
    cudaMemcpy(&sum, d_tmp, sizeof(ll), cudaMemcpyDeviceToHost);

    printf("sum = %llu", sum);

}