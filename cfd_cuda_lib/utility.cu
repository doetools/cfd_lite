#define N_BLOCK_1D 32
#define N_BLOCK_2D 1024
#define RED 1
#define GREEN 2

#define ix(i, j) ((j) + (row_size) * (i))
#define ix_i(gid) (gid / row_size)
#define ix_j(gid) (gid % row_size)

#define LEGAL 1

__global__ void reduce(float *x, const int x_size, float *y, const int y_size) {

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

  for (unsigned int i = block_dim / 2; i > 0; i /= 2) {
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

__global__ void gs_colors(int *colors, const int row_size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int i = ix_i(gid);
  int j = ix_j(gid);

  if (i % 2 == 0) {
    if (j % 2 == 0)
      colors[gid] = RED;
    else
      colors[gid] = GREEN;
  }

  if (i % 2 == 1) {
    if (j % 2 == 0)
      colors[gid] = GREEN;
    else
      colors[gid] = RED;
  }
}

__global__ void gauss_seidel(float *x, float *a_e, float *a_w, float *a_n,
                             float *a_s, float *a_p, float *b, const float sor,
                             int *type, int *colors, int target_color,
                             const int x_size, const int row_size) {
  int block_x = blockIdx.x;
  int block_dim = blockDim.x;
  int lid = threadIdx.x;

  int gid = block_x * block_dim + lid;
  int i = ix_i(gid);
  int j = ix_j(gid);

  float tmp;

  if (gid < x_size && colors[gid] == target_color && type[gid] == LEGAL) {
    tmp = (a_e[gid] * x[ix(i, j + 1)] + a_w[gid] * x[ix(i, j - 1)] +
           a_n[gid] * x[ix(i + 1, j)] + a_s[gid] * x[ix(i - 1, j)] + b[gid]) /
          a_p[gid];

    x[gid] = sor * tmp + (1 - sor) * x[gid];
  }
}