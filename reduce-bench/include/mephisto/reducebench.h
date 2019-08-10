#include <cassert>
#include <numeric>
#include <random>

#include <dash/Array.h>
#include <dash/algorithm/Generate.h>
#include <dash/algorithm/LocalRange.h>
#include <dash/algorithm/Transform.h>

#include <mephisto/execution>

#include <util/Logging.h>

template <typename RandomIt, typename Gen>
inline void parallel_rand(RandomIt begin, RandomIt end, Gen const g)
{
  assert(!(end < begin));

  auto const n = static_cast<size_t>(std::distance(begin, end));

  auto const l_range = dash::local_index_range(begin, end);

  using pointer = typename std::iterator_traits<RandomIt>::pointer;

  auto* lbegin = dash::local_begin(
      static_cast<pointer>(begin), begin.pattern().team().myid());
  auto*      lend   = std::next(lbegin, l_range.end);
  auto const nl     = l_range.end - l_range.begin;

  auto const myid = begin.pattern().team().myid();

  std::seed_seq seed{std::random_device{}(), static_cast<unsigned>(myid)};
  std::mt19937  rng(seed);

  for (size_t idx = 0; idx < nl; ++idx) {
    auto it = lbegin + idx;
    *it     = g(n, idx, rng);
  }

  begin.pattern().team().barrier();
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
  auto end   = c.end();
  assert(!(end < begin));

  // Context consists of the host, the accelerator and the stream
  using Queue   = alpaka::queue::QueueCpuSync;
  using EntityT =
      mephisto::Entity<1, std::size_t, alpaka::acc::AccCpuSerial>;
  using Context = mephisto::execution::AlpakaExecutionContext<EntityT, Queue>;
  Context ctx;

  // The executor is the one actually doing the computation
  mephisto::execution::AlpakaExecutor<Context> executor{&ctx};

  // The policy is used to relax guarantees.
  auto policy = mephisto::execution::make_parallel_policy(executor);

  return dash::transform_reduce(policy, begin, end, init, binary_op, unary_op);
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
  assert(!(end < begin));

  using value_t  = typename dash::iterator_traits<RandomIt>::value_type;
  using result_t = ResultT;

  auto const l_range = dash::local_index_range(begin, end);

  using pointer = typename std::iterator_traits<RandomIt>::pointer;

  auto* lbegin = dash::local_begin(
      static_cast<pointer>(begin), begin.pattern().team().myid());
  auto*      lend = std::next(lbegin, l_range.end);
  auto const nl   = l_range.end - l_range.begin;

  std::transform(lbegin, lend, lbegin, unary_op);
  auto const local_result = std::accumulate(lbegin, lend, init, binary_op);

  auto const myid    = begin.pattern().team().myid();
  auto       results = dash::Array<result_t>(begin.pattern().team().size());
  results[myid]      = local_result;

  results.barrier();

  result_t global_result{init};
  for(auto val : results) {
    global_result = binary_op(global_result, val);
  }

  begin.pattern().team().barrier();
  return global_result == to_verify;
}
