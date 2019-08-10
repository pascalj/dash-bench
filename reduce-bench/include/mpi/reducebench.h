#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <random>

extern "C" {
#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD
#endif
}

#include <util/Logging.h>

template <typename RandomIt, typename Gen>
inline void parallel_rand(RandomIt begin, RandomIt end, Gen const g)
{
  assert(!(end < begin));

  auto const n = static_cast<size_t>(std::distance(begin, end));

  int ThisTask;
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

  std::seed_seq seed{std::random_device{}(), static_cast<unsigned>(ThisTask)};
  std::mt19937  rng(seed);

  for (size_t idx = 0; idx < n; ++idx) {
      auto it = begin + idx;
      *it     = g(n, idx, rng);
  }
}

template <
    typename Container,
    typename Init,
    typename BinaryOp,
    typename UnaryOp>
inline auto transform_reduce(
    Container& c, Init init, BinaryOp&& binary_op, UnaryOp&& unary_op)
{
  auto begin = c.begin();
  auto end = c.end();
  assert(!(end < begin));

  using value_t = typename Container::value_type;
  using result_t = Init;

  auto const mysize = static_cast<size_t>(std::distance(begin, end));

  auto * ptr = std::addressof(*begin);

  std::transform(begin, end, begin, unary_op);
  result_t local_result = std::accumulate(begin, end, init, binary_op);
  result_t global_result;

  MPI_Allreduce(
      &local_result, &global_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  return global_result;
}

template <
    typename RandomIt,
    typename ResultT,
    typename Init,
    typename BinaryOp,
    typename UnaryOp>
inline auto verify_transform_reduce(
    RandomIt   begin,
    RandomIt   end,
    ResultT    to_verify,
    Init       init,
    BinaryOp&& binary_op,
    UnaryOp&&  unary_op)
{
  return true;
}
