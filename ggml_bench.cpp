#include "common.h"
#include "benchmark.h"

int main(int argc, char ** argv) {
    benchmark_params bparams;
    if (!benchmark_params_parse(argc, argv, bparams)) {
        return 1;
    }

    run_benchmark(bparams);

    return 0;
}
