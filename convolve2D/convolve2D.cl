__kernel void convolve2D(__global double *matrix,
                         __global double *convolution_kernel,
                         __global double *result,
                         int n, int m) {
  int row = get_global_id(0);
  int col = get_global_id(1);

  if (row >= n || col >= n) {
    return;
  }

  double sum = 0;

  int half_m = (m - 1) / 2;

  for (int i = -half_m; i <= half_m; ++i) {
    for (int j = -half_m; j <= half_m; ++j) {
      if (row + i < 0 || col + j < 0 || row + i >= n || col + j >= n)
        continue;

      sum += matrix[(row + i) * n + (col + j)] *
             convolution_kernel[(i + half_m) * m + (j + half_m)];
    }
  }

  result[row * n + col] = sum;
}