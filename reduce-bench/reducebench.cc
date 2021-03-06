#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

#if defined(USE_TBB_HIGHLEVEL) || defined(USE_TBB_LOWLEVEL)
#include <tbb/sortbench.h>
#include <tbb/task_scheduler_init.h>
#elif defined(USE_OPENMP)
#include <openmp/sortbench.h>
#elif defined(USE_DASH)
#ifdef DASH_ENABLE_PSTL
#include <tbb/task_scheduler_init.h>
#endif

#include <dash/util/BenchmarkParams.h>
#include <dash/reducebench.h>
#include <omp.h>
#include <libdash.h>
#elif defined(USE_MPI)
#include <mpi/reducebench.h>
#elif defined(USE_MEPHISTO)
#include <dash/util/BenchmarkParams.h>
#include <libdash.h>
#include <patterns/local_pattern.h>
#include <alpaka/alpaka.hpp>
#include <mephisto/reducebench.h>
#endif

#include <util/Logging.h>
#include <util/Random.h>
#include <util/Timer.h>

#define GB (1 << 30)
#define MB (1 << 20)

#ifndef FN_HOST
#define FN_HOST
#endif

#ifndef FN_HOST_ACC
#define FN_HOST_ACC
#endif


template <typename RandomIt>
void trace_histo(RandomIt begin, RandomIt end)
{
#ifdef ENABLE_LOGGING
  auto const              n = std::distance(begin, end);
  std::map<key_t, size_t> hist{};
  for (size_t idx = 0; idx < n; ++idx) {
    ++hist[begin[idx]];
  }

  for (auto p : hist) {
    LOG(p.first << ' ' << p.second);
  }
#endif
}

static constexpr size_t BURN_IN = 1;
static constexpr size_t NITER   = 10;

void print_header(std::string const& app, double mb, int P)
{
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++\n";
  std::cout << "++              Reduce Bench                   ++\n";
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++\n";
  std::cout << std::setw(20) << "NTasks: " << P << "\n";
  std::cout << std::setw(20) << "Size: " << std::fixed << std::setprecision(2)
            << mb << "\n";
#if defined(USE_DASH) || defined(USE_MPI) || defined(USE_USORT) || defined(USE_MEPHISTO)
  std::cout << std::setw(20) << "Size per Unit (MB): " << std::fixed
            << std::setprecision(2) << mb / P;
#endif
  std::cout << "\n\n";
  // Print the header
  std::cout << std::setw(4) << "#,";
  std::cout << std::setw(10) << "NTasks,";
  std::cout << std::setw(10) << "ThreadsPerTask,";
  std::cout << std::setw(10) << "Size (MB),";
  std::cout << std::setw(20) << "Time,";
  std::cout << std::setw(20) << "Test Case";
  std::cout << "\n";
}

//! Test sort for n items
template <class Container>
void Test(Container & c, size_t N, int r, size_t P, size_t threads, std::string const& test_case)
{
  using key_t = typename Container::value_type;

  auto const mb = N * sizeof(key_t) / MB;

  using dist_t = sortbench::NormalDistribution<key_t>;

  //static dist_t dist{key_t{0}, key_t{(1 << 20)}};
  static dist_t dist{};

#if defined(USE_DASH) || defined(USE_MEPHISTO)

  constexpr int nSamples = 250;

  // The DASH Trace does not really scale, so we select at most nSamples units
  // which trace
  std::vector<dash::team_unit_t> trace_unit_samples(nSamples);
  int                            id_stride = P / nSamples;
  if (id_stride < 2) {
    std::iota(
        std::begin(trace_unit_samples), std::end(trace_unit_samples), 0);
  }
  else {
    dash::team_unit_t v_init{0};
    std::generate(
        std::begin(trace_unit_samples),
        std::end(trace_unit_samples),
        [&v_init, id_stride]() {
          auto val = v_init;
          v_init += id_stride;
          return val;
        });
  }

#endif

  for (size_t iter = 0; iter < NITER + BURN_IN; ++iter) {
    parallel_rand(
        c.begin(), c.end(), [](size_t total, size_t index, std::mt19937& rng) {
          return index;
        });

#ifdef USE_DASH
    dash::util::TraceStore::on();
    dash::util::TraceStore::clear();
#endif

    using value_t = uint64_t;
    value_t init{0};

    auto binary_op = [] FN_HOST_ACC (value_t lhs, value_t rhs) {
      return lhs + rhs;
    };

    auto unary_op = [] FN_HOST_ACC (value_t val) {
      // some light computing, so this is not a memory benchmark
      return val % (1 + val * val);
    };

    auto const start    = ChronoClockNow();
    auto const result   = transform_reduce(c, init, binary_op, unary_op);
    auto const duration = ChronoClockNow() - start;

    auto const ret = verify_transform_reduce(
        c.begin(), c.end(), result, init, binary_op, unary_op);

    if (ret == 0) {
      std::cerr << "validation failed! (n = " << N << "): " << result
                << std::endl;
    }

    if (iter >= BURN_IN && r == 0) {
      std::ostringstream os;
      // Iteration
      os << std::setw(3) << iter << ",";
      // Ntasks
      os << std::setw(9) << P << ",";
      os << std::setw(9) << threads << ",";
      // Size
      os << std::setw(9) << std::fixed << std::setprecision(2) << mb;
      os << ",";
      // Time (s)
      os << std::setw(19) << std::fixed << std::setprecision(8);
      os << duration << ",";
      // Test Case
      os << std::setw(20) << test_case;
      os << "\n";
      std::cout << os.str();
    }

#ifdef USE_DASH
    c.begin().pattern().team().barrier();
    if (iter == (NITER + BURN_IN - 1) &&
        // if the id of this task is included in samples
        (std::find(
             std::begin(trace_unit_samples),
             std::end(trace_unit_samples),
             dash::team_unit_t{r}) != std::end(trace_unit_samples))) {
      dash::util::TraceStore::write(std::cout, false);
    }

    dash::util::TraceStore::off();
    dash::util::TraceStore::clear();
#endif
  }
}

int main(int argc, char* argv[])
{
  using key_t = uint64_t;

  if (argc < 2) {
    std::cout << std::string(argv[0])
#if defined(USE_DASH) || defined(USE_MPI) || defined(USE_USORT) || defined(USE_MEPHISTO)
              << " [nbytes per rank]\n";
#else
              << " [nbytes]\n";
#endif
    return 1;
  }

  // Size in Bytes
  auto const mysize = static_cast<size_t>(atoll(argv[1]));
  // Number of local elements
  auto const nl = mysize / sizeof(key_t);
  // Number of threads
  auto const T =
      (argc == 3) ? atoi(argv[2]) : 0;

#if defined(USE_DASH) || defined(USE_MEPHISTO)
  dash::init(&argc, &argv);
  auto const P           = dash::size();
  auto const gsize_bytes = mysize * P;
  auto const N           = nl * P;
  auto const r           = dash::myid();
#elif defined(USE_MPI) || defined(USE_USORT)
  MPI_Init(&argc, &argv);
  int P;
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  auto const gsize_bytes = mysize * P;
  auto const N           = nl * P;
  int        r;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
#else
  auto const P =
      T ? T : std::thread::hardware_concurrency();
  assert(P > 0);
  auto const gsize_bytes = mysize;

  auto const N           = nl;
  auto const r    = 0;
#endif

  size_t threads = 1;
  auto thread_var = std::getenv("THREADS");
  if(thread_var != NULL) {
  	threads = std::atoi(thread_var);
  }
#if defined(_OPENMP)
  omp_set_num_threads(threads);
#endif


#if defined(USE_TBB_HIGHLEVEL) || defined(USE_TBB_LOWLEVEL)
  tbb::task_scheduler_init init{static_cast<int>(P)};
#elif defined(USE_DASH) && defined(DASH_ENABLE_PSTL)
  tbb::task_scheduler_init init{omp_get_max_threads()};
#endif

#if defined(USE_OPENMP)
  omp_set_num_threads(P);
#endif

  double mb = (gsize_bytes / MB);

  std::string const executable(argv[0]);
  auto const        base_filename =
      executable.substr(executable.find_last_of("/\\") + 1);

#if defined(USE_MEPHISTO)
  /* using BasePattern = dash::BlockPattern<1>; */
  using BasePattern = dash::TilePattern<1>;
  using PatternT    = patterns::BalancedLocalPattern<BasePattern, entity_t<1>>;

  dash::DistributionSpec<1> distspec(dash::TILE(N / 2 / dash::size()));
  BasePattern base{N, distspec};
  PatternT pattern{base};

  dash::Array<key_t, dash::default_index_t, PatternT, memory_t> keys(pattern);
#elif defined(USE_DASH)
  dash::Array<key_t> keys(N);
#else
  std::vector<key_t> keys(nl);
#endif

  if (r == 0) {
#if defined(USE_DASH) || defined(USE_MEPHISTO)
    dash::util::BenchmarkParams bench_params("bench.mephisto.reduce");
    bench_params.set_output_width(72);
    bench_params.print_header();
    if (dash::size() < 200) {
      bench_params.print_pinning();
    }
#endif
    print_header(base_filename, mb, P);
  }

  Test(keys, N, r, P, threads, base_filename);

#if defined(USE_DASH) || defined(USE_MEPHISTO)
  dash::finalize();
#elif defined(USE_MPI) || defined(USE_USORT)
  MPI_Finalize();
#endif

  if (r == 0) {
    std::cout << "\n";
  }

  return 0;
}
