#define SWAP(a, b) {__local double * tmp = a; a = b; b = tmp;}

__kernel void prefix_sum(__global double *input,
                         __global double *output,
                         __local double *prev_sum,
                         __local double *next_sum,
                         int size) {
  uint global_id = get_global_id(0);
  uint local_id = get_local_id(0);
  uint block_size = get_local_size(0);

  if (global_id < size) {
    prev_sum[local_id] = next_sum[local_id] = input[global_id];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint s = 1; s < block_size; s <<= 1) {
    if (local_id > (s - 1)) {
      next_sum[local_id] = prev_sum[local_id] + prev_sum[local_id - s];
    } else {
      next_sum[local_id] = prev_sum[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    SWAP(prev_sum, next_sum);
  }

  if (global_id < size) {
    output[global_id] = prev_sum[local_id];
  }
}

__kernel void add_to_every_block(__global double *input,
                                 __global double *sums_input,
                                 __global double *output,
                                 int size) {
  uint global_id = get_global_id(0);
  uint block_size = get_local_size(0);

  if (global_id < size) {
    output[global_id] = input[global_id] + sums_input[global_id / block_size];
  }
}