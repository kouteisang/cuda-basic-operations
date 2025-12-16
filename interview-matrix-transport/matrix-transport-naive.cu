#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace std;

// #define BLK_NUM 2
// #define BLK_DIM 128


template<typename T>
__global__ void transport(T *d_in, T *d_out, int h, int w){

    int tx = blockDim.x * blockIdx.x + threadIdx.x; // col
    int ty = blockDim.y * blockIdx.y + threadIdx.y; // row
    if(tx < w && ty < h){
        d_out[tx * h + ty] = d_in[ty * w + tx];
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
    dim3 blk_num((w+16-1)/16, (h+16-1)/16);
    dim3 blk_dim(16, 16);
    transport<<<blk_num, blk_dim>>>(d_in, d_out, h, w);
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
    for(int i = 1; i < argc; i ++){
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