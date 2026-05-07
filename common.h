#pragma once

#include "ggml.h"
#include <string>

enum class benchmark_op {
    ADD,
    SUB,
    MUL,
    DIV
};

struct benchmark_params {
    ggml_type type = GGML_TYPE_F32;
    int matrix_size = 256;
    benchmark_op op = benchmark_op::ADD;
};

bool benchmark_params_parse(int argc, char ** argv, benchmark_params & params);
void benchmark_params_print_usage(char ** argv);
std::string benchmark_params_get_type_str(const benchmark_params & params);
std::string benchmark_params_get_op_str(const benchmark_params & params);
