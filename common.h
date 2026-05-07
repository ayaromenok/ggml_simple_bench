#pragma once

#include "ggml.h"
#include <string>

struct benchmark_params {
    ggml_type type = GGML_TYPE_F32;
    int matrix_size = 256;
};

bool benchmark_params_parse(int argc, char ** argv, benchmark_params & params);
void benchmark_params_print_usage(char ** argv);
std::string benchmark_params_get_type_str(const benchmark_params & params);
