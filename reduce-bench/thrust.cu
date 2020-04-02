#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <chrono>

int main(int argc, char** argv)
{
  if (argc != 2) {
	  printf("Usage: %s [n bytes]\n", argv[0]);
	  return 1;
  } 

  size_t nbytes = std::stoll(argv[1]);
  thrust::host_vector<long long> h_vec(nbytes / sizeof(long long));
  std::generate(h_vec.begin(), h_vec.end(), rand);

  auto start = std::chrono::high_resolution_clock::now();
  // transfer data to the device
  thrust::device_vector<long long> d_vec = h_vec;

  auto unary_op = [] __device__ (auto val) {
	  return val % (1 + val * val);
  };
  auto binary_op = [] __device__ (auto acc, auto in) {
	  return acc + in;
  };
  long long init = 0;
  auto result = thrust::transform_reduce(d_vec.begin(), d_vec.end(), unary_op, init, binary_op);
  auto end = std::chrono::high_resolution_clock::now();

  // transfer data back to host
  // thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  fprintf(stderr, "result: %lld\n", result);
  printf("1, 1, 1, %ld, %f, thrust\n", nbytes / (1024 * 1024), duration);

  return 0;
}

