#include <cuda_runtime.h>
#include <iostream>

using namespace std;

template<typename T>
__global__ void double_data(T* d_data, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for(int t = tid; t < n; t += step){
        d_data[t] = d_data[t] * 2;
    }
}


int main(){
    using T = int;
    int n = 2<< 5;
    T* data;
    data = new T[n];

    for(int i = 0; i < n; i ++){
        data[i] = (T)i;
    }

    T* d_data;
    cudaMalloc(&d_data, sizeof(T) * n);

    cudaMemcpy(d_data, data, sizeof(T) * n, cudaMemcpyHostToDevice);

    double_data<T><<<256, 1024>>>(d_data, n);

    cudaMemcpy(data, d_data, sizeof(T) * n, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i ++){
        std::cout << data[i] << " ";
    }

}