#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace std;

#define TILE 32

template<typename T>
__global__ void transport_tile(T* __restrict__ d_in, T* __restrict__ d_out, int h, int w){
    
    __shared__ T tile[TILE][TILE + 1]; // avoid bank conflict
    
    int row = blockIdx.y * TILE;
    int col = blockIdx.x * TILE;

    for(int r = threadIdx.y; r < TILE; r += blockDim.y){
        int rr = r + row;
        for(int c = threadIdx.x; c < TILE; c += blockDim.x){
            int cc = c + col;
            if(rr < h && cc < w){
                tile[r][c] = d_in[w * rr+ cc];
            }
        } 
    }
    __syncthreads();

    int out_row0 = blockIdx.x * TILE;
    int out_col0 = blockIdx.y * TILE;

    for(int r = threadIdx.y; r < TILE; r += blockDim.y){
        int rr = out_row0 + r;
        for(int c = threadIdx.x; c < TILE; c += blockDim.x){
            int cc = out_col0 + c;
            if(rr < w && cc < h){
                d_out[rr*h + cc] = tile[c][r];
            }
        }
    }
}


template<typename T>
void run(int h, int w){

    vector<T> h_in(h*w), h_out(h*w);
    for(int i = 0; i < h*w; i ++){
        h_in[i] = i*1.0;
    }

    T *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(T) * h * w);
    cudaMalloc(&d_out, sizeof(T) * h * w);
    cudaMemcpy(d_in, h_in.data(), sizeof(T) * h * w, cudaMemcpyHostToDevice);
    dim3 blk_num((w+TILE-1)/TILE, (h+TILE-1)/TILE);
    dim3 blk_dim(8, 8);
    transport_tile<<<blk_num, blk_dim>>>(d_in, d_out, h, w);
    cudaMemcpy(h_out.data(), d_out, sizeof(T) * h * w, cudaMemcpyDeviceToHost);
    for(int i = 0; i < w; i ++){
        for(int j = 0; j < h; j ++){
            printf("%f ", h_out[i*h + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv){
    
    int h = 0;
    int w = 0;
    bool has_h = false;
    bool has_w = false;
    for(int i = 0; i < argc; i ++){
        string arg = argv[i];
        if(arg == "-h"){
            h = stoi(argv[++i]);
            has_h = true;
        }else if(arg == "-w"){
            w = stoi(argv[++i]);
            has_w = true;
        }
    }

    if(!has_h || !has_w){
        printf("Parameter Error\n");
        return 0;
    }


    printf("h = %d, w = %d\n", h, w);
    

    run<float>(h, w);


}