#include "common.h"
#include <cstdio>
#include <cstring>
#include <string>

void benchmark_params_print_usage(char ** argv) {
    fprintf(stderr, "Usage: %s [-t|--type f16|f32] [-ms|--matrix-size size] [-mt|--matrix-type 1|2|3] [-o|--operator add|sub|mul|div]\n", argv[0]);
}

bool benchmark_params_parse(int argc, char ** argv, benchmark_params & params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-t" || arg == "--type") {
            if (++i < argc) {
                std::string val = argv[i];
                if (val == "f32" || val == "fp32") {
                    params.type = GGML_TYPE_F32;
                } else if (val == "f16" || val == "fp16") {
                    params.type = GGML_TYPE_F16;
                } else {
                    fprintf(stderr, "error: unknown type '%s'\n", val.c_str());
                    return false;
                }
            } else {
                fprintf(stderr, "error: %s requires an argument\n", arg.c_str());
                return false;
            }
        } else if (arg == "-ms" || arg == "--matrix-size") {
            if (++i < argc) {
                try {
                    params.matrix_size = std::stoi(argv[i]);
                } catch (const std::exception & e) {
                    fprintf(stderr, "error: invalid matrix size '%s'\n", argv[i]);
                    return false;
                }
            } else {
                fprintf(stderr, "error: %s requires an argument\n", arg.c_str());
                return false;
            }
        } else if (arg == "-mt" || arg == "--matrix-type") {
            if (++i < argc) {
                try {
                    params.matrix_type = std::stoi(argv[i]);
                    if (params.matrix_type < 1 || params.matrix_type > 3) {
                        fprintf(stderr, "error: matrix type must be 1, 2, or 3\n");
                        return false;
                    }
                } catch (const std::exception & e) {
                    fprintf(stderr, "error: invalid matrix type '%s'\n", argv[i]);
                    return false;
                }
            } else {
                fprintf(stderr, "error: %s requires an argument\n", arg.c_str());
                return false;
            }
        } else if (arg == "-o" || arg == "--operator") {
            if (++i < argc) {
                std::string val = argv[i];
                if (val == "add") {
                    params.op = benchmark_op::ADD;
                } else if (val == "sub") {
                    params.op = benchmark_op::SUB;
                } else if (val == "mul") {
                    params.op = benchmark_op::MUL;
                } else if (val == "div") {
                    params.op = benchmark_op::DIV;
                } else {
                    fprintf(stderr, "error: unknown operator '%s'\n", val.c_str());
                    return false;
                }
            } else {
                fprintf(stderr, "error: %s requires an argument\n", arg.c_str());
                return false;
            }
        } else {
            benchmark_params_print_usage(argv);
            return false;
        }
    }
    return true;
}

std::string benchmark_params_get_op_str(const benchmark_params & params) {
    switch (params.op) {
        case benchmark_op::ADD: return "add";
        case benchmark_op::SUB: return "sub";
        case benchmark_op::MUL: return "mul";
        case benchmark_op::DIV: return "div";
        default: return "unknown";
    }
}

std::string benchmark_params_get_type_str(const benchmark_params & params) {
    if (params.type == GGML_TYPE_F32) return "f32";
    if (params.type == GGML_TYPE_F16) return "f16";
    return "unknown";
}
