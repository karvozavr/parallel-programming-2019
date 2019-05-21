#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>


static const std::string KERNEL_SRC = "prefix-sum.cl";
static const size_t BLOCK_SIZE = 256;

int align_to_block_size(int size, int block_size) {
  return ((size + block_size - 1) / block_size) * block_size;
}

void add_to_every_block(std::vector<double> const &data,
                        std::vector<double> const &block_add,
                        std::vector<double> &output,
                        cl::Context &context,
                        cl::CommandQueue &command_queue,
                        cl::Program const &program) {
  size_t data_bytes_size = data.size() * sizeof(double);
  size_t block_add_bytes_size = block_add.size() * sizeof(double);
  int block_aligned_size = align_to_block_size(data.size(), BLOCK_SIZE);

  cl::Buffer dev_in(context, CL_MEM_READ_ONLY, data_bytes_size);
  cl::Buffer dev_block_in(context, CL_MEM_READ_ONLY, data_bytes_size);
  cl::Buffer dev_out(context, CL_MEM_WRITE_ONLY, data_bytes_size);

  // copy from cpu to device
  command_queue.enqueueWriteBuffer(dev_in, CL_TRUE, 0, data_bytes_size, data.data());
  command_queue.enqueueWriteBuffer(dev_block_in, CL_TRUE, 0, block_add_bytes_size, block_add.data());

  // load named kernel from opencl source
  cl::Kernel kernel(program, "add_to_every_block");
  cl::KernelFunctor prefix_sum_functor(kernel,
                                       command_queue,
                                       cl::NullRange,
                                       cl::NDRange(block_aligned_size),
                                       cl::NDRange(BLOCK_SIZE));
  prefix_sum_functor(dev_in,
                     dev_block_in,
                     dev_out,
                     static_cast<int>(data.size()));

  // copy from device to cpu
  command_queue.enqueueReadBuffer(dev_out, CL_TRUE, 0, data_bytes_size, output.data());
}

void prefix_sum(std::vector<double> const &data,
                std::vector<double> &output,
                cl::Context &context,
                cl::CommandQueue &command_queue,
                cl::Program const &program) {
  size_t data_bytes_size = data.size() * sizeof(double);
  int block_aligned_size = align_to_block_size(data.size(), BLOCK_SIZE);

  cl::Buffer dev_in(context, CL_MEM_READ_ONLY, data_bytes_size);
  cl::Buffer dev_out(context, CL_MEM_WRITE_ONLY, data_bytes_size);

  // copy from cpu to device
  command_queue.enqueueWriteBuffer(dev_in, CL_TRUE, 0, data_bytes_size, data.data());

  // load named kernel from opencl source
  cl::Kernel kernel(program, "prefix_sum");
  cl::KernelFunctor prefix_sum_functor(kernel,
                                       command_queue,
                                       cl::NullRange,
                                       cl::NDRange(block_aligned_size),
                                       cl::NDRange(BLOCK_SIZE));
  prefix_sum_functor(dev_in,
                     dev_out,
                     cl::__local(BLOCK_SIZE * sizeof(double)),
                     cl::__local(BLOCK_SIZE * sizeof(double)),
                     static_cast<int>(data.size()));

  // copy from device to cpu
  command_queue.enqueueReadBuffer(dev_out, CL_TRUE, 0, data_bytes_size, output.data());

  // if array doesnt fit to single block, we need to propagate partial sums to further blocks
  if (data.size() > BLOCK_SIZE) {
    size_t blocks = ((data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    std::vector<double> block_sums(blocks);
    double running_sum = 0;
    for (int i = 1; i < blocks; ++i) {
      running_sum += output[BLOCK_SIZE * i - 1];
      block_sums[i] = running_sum;
    }

    add_to_every_block(output, block_sums, output, context, command_queue, program);
  }
}

int main() {
  std::freopen("input.txt", "r", stdin);
  std::freopen("output.txt", "w", stdout);

  size_t N = 0;

  std::cin >> N;

  std::vector<double> input(N);
  std::vector<double> output(N);

  for (int i = 0; i < N; ++i) {
    std::cin >> input[i];
  }

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
    cl::Program program(context, source);

    // compile opencl source
    size_t const block_size = 16;
    std::string block_size_option = "-D BLOCK_SIZE=" + std::to_string(block_size);
    program.build(devices, block_size_option.c_str());

    // calculate prefix sum
    prefix_sum(input, output, context, command_queue, program);
  } catch (cl::Error &e) {
    std::cerr << std::endl << "OpenCL error: " << e.what() << " : " << e.err() << std::endl;
  }

  std::cout << std::fixed << std::setprecision(3);

  for (double x : output) {
    std::cout << x << ' ';
  }

  return 0;
}