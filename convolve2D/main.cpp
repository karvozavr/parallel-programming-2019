#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

template<typename T>
void fill_matrix_from_stdin(T *matrix, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t idx = i * N + j;
      std::cin >> matrix[idx];
    }
  }
}

template<typename T>
void fill_matrix(T *matrix, size_t N, T value) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t idx = i * N + j;
      matrix[idx] = value;
    }
  }
}

static const std::string KERNEL_SRC = "convolve2D.cl";

int main() {
  std::freopen("input.txt", "r", stdin);
  std::freopen("output.txt", "w", stdout);

  size_t N = 0;
  size_t M = 0;

  std::cin >> N >> M;

  size_t matrix_size = N * N;
  size_t kernel_size = M * M;

  std::unique_ptr<double[]> matrix(new double[matrix_size]);
  std::unique_ptr<double[]> kernel(new double[kernel_size]);
  std::unique_ptr<double[]> result(new double[matrix_size]);

  fill_matrix_from_stdin<double>(matrix.get(), N);
  fill_matrix_from_stdin<double>(kernel.get(), M);
  fill_matrix<double>(result.get(), N, 0);

  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  std::vector<cl::Kernel> kernels;

  try {
    // create platform
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // create context
    cl::Context context(devices);

    // create command queue
    cl::CommandQueue command_queue(context, devices[0]);

    // load opencl source
    std::ifstream cl_file(KERNEL_SRC);
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

    // create program
    cl::Program convolve2D_program(context, source);

    // compile opencl source
    size_t const block_size = 16;
    std::string block_size_option = "-D BLOCK_SIZE=" + std::to_string(block_size);
    convolve2D_program.build(devices, block_size_option.c_str());

    // allocate device buffers
    size_t matrix_buffer_size = sizeof(double) * matrix_size;
    size_t kernel_buffer_size = sizeof(double) * kernel_size;

    cl::Buffer dev_matrix(context, CL_MEM_READ_ONLY, matrix_buffer_size);
    cl::Buffer dev_kernel(context, CL_MEM_READ_ONLY, kernel_buffer_size);
    cl::Buffer dev_result(context, CL_MEM_WRITE_ONLY, matrix_buffer_size);

    // copy from cpu to device
    command_queue.enqueueWriteBuffer(dev_matrix, CL_TRUE, 0, matrix_buffer_size, matrix.get());
    command_queue.enqueueWriteBuffer(dev_kernel, CL_TRUE, 0, kernel_buffer_size, kernel.get());

    // load named kernel from opencl source
    cl::Kernel convolve2D_kernel(convolve2D_program, "convolve2D");
    cl::KernelFunctor convolve2D_functor(convolve2D_kernel, command_queue, cl::NullRange,
                                         cl::NDRange(((N + block_size - 1) / block_size) * block_size,
                                                     ((N + block_size - 1) / block_size) * block_size),
                                         cl::NDRange(block_size, block_size));
    convolve2D_functor(dev_matrix, dev_kernel, dev_result, (int) N, (int) M);

    // copy from device to cpu
    command_queue.enqueueReadBuffer(dev_result, CL_TRUE, 0, matrix_buffer_size, result.get());

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        std::cout << result.get()[idx] << " ";
      }
      std::cout << std::endl;
    }

  } catch (cl::Error &e) {
    std::cerr << std::endl << "OpenCL error: " << e.what() << " : " << e.err() << std::endl;

  }

  return 0;
}
