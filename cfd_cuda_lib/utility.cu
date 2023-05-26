#define N_BLOCK_1D 32
#define N_BLOCK_2D 1024

__global__ void reduce(float *x, const int x_size, float *y, const int y_size)
{

    // find cuda kernel id.
    int block_x = blockIdx.x;
    int block_dim = blockDim.x;
    int local_id = threadIdx.x;
    int global_id = block_x * block_dim + local_id;

    // declare a shared memory
    __shared__ float tmp[N_BLOCK_2D];

    // read from global to shared memory
    if (global_id < x_size)
        tmp[local_id] = x[global_id];
    else
        tmp[local_id] = 0.0;

    // wating unitl reading global data to shared memory by block
    __syncthreads();

    for (unsigned int i = block_dim / 2; i > 0; i /= 2)
    {
        // add through half of the size
        if (local_id < i)
            tmp[local_id] += tmp[local_id + i];

        // wait until all threads done before moving to next sweep
        __syncthreads();
    }

    // save
    if (block_x < y_size)
        y[block_x] = tmp[0];
}
