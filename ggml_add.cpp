#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>
#include <algorithm>
#include <string>

int main(int argc, char ** argv) {
    ggml_type type = GGML_TYPE_F32;
    std::string type_str = "f32";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-t" || arg == "--type") {
            if (++i < argc) {
                std::string val = argv[i];
                if (val == "f32"||val=="fp32") {
                    type = GGML_TYPE_F32;
                    type_str = "f32";
                } else if (val == "f16"||val=="fp16") {
                    type = GGML_TYPE_F16;
                    type_str = "f16";
                } else {
                    fprintf(stderr, "error: unknown type '%s'\n", val.c_str());
                    return 1;
                }
            } else {
                fprintf(stderr, "error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else {
            fprintf(stderr, "Usage: %s [-t|--type f16|f32]\n", argv[0]);
            return 1;
        }
    }

    const int nx = 1024;
    const int ny = 1024;
    const int nz = 512;

    std::cout << "Running GGML CUDA Add benchmark (" << type_str << ")..." << std::endl;

    size_t ctx_size = 0;
    {
        ctx_size += 3 * ggml_tensor_overhead(); // tensors a, b, c
        ctx_size += ggml_graph_overhead();      // compute graph
        ctx_size += 1024;                       // some overhead
    }

    // 1. Initialize GGML context
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 2. Create tensors
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, type, nx, ny, nz);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, type, nx, ny, nz);
    struct ggml_tensor * c = ggml_add(ctx, a, b);
    
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    ggml_set_name(c, "c");

    // 3. Initialize CUDA backend
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init() failed\n");
        return 1;
    }

    // 4. Allocate tensors on the backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "ggml_backend_alloc_ctx_tensors() failed\n");
        return 1;
    }

    // 5. Fill tensors with random values (Parallel)
    size_t total_elements = (size_t)nx * ny * nz;
    std::vector<char> h_a_raw(ggml_nbytes(a));
    std::vector<char> h_b_raw(ggml_nbytes(b));

    std::cout << "Filling matrices with 16 predefined values (parallel on CPU)..." << std::endl;

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    size_t chunk_size = (total_elements + num_threads - 1) / num_threads;

    auto fill_task = [&](size_t start, size_t end, int thread_id) {
        std::mt19937 local_rng(42 + thread_id);
        std::uniform_int_distribution<int> local_dist(0, 15);
        
        if (type == GGML_TYPE_F16) {
            ggml_fp16_t * h_a = (ggml_fp16_t *)h_a_raw.data();
            ggml_fp16_t * h_b = (ggml_fp16_t *)h_b_raw.data();
            std::vector<ggml_fp16_t> predefined(16);
            for (int i = 0; i < 16; ++i) predefined[i] = ggml_fp32_to_fp16(0.125f * (i + 1));
            
            for (size_t i = start; i < end; ++i) {
                h_a[i] = predefined[local_dist(local_rng)];
                h_b[i] = predefined[local_dist(local_rng)];
            }
        } else {
            float * h_a = (float *)h_a_raw.data();
            float * h_b = (float *)h_b_raw.data();
            std::vector<float> predefined(16);
            for (int i = 0; i < 16; ++i) predefined[i] = 0.125f * (i + 1);
            
            for (size_t i = start; i < end; ++i) {
                h_a[i] = predefined[local_dist(local_rng)];
                h_b[i] = predefined[local_dist(local_rng)];
            }
        }
    };

    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, total_elements);
        if (start < total_elements) {
            threads.emplace_back(fill_task, start, end, (int)t);
        }
    }

    for (auto & thread : threads) thread.join();

    std::cout << "Uploading to GPU..." << std::endl;
    ggml_backend_tensor_set(a, h_a_raw.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, h_b_raw.data(), 0, ggml_nbytes(b));

    // 6. Define the computation graph
    struct ggml_init_params params_cgraph = {
        /* .mem_size   = */ ggml_graph_overhead() + 1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx_cgraph = ggml_init(params_cgraph);
    struct ggml_cgraph * gf = ggml_new_graph(ctx_cgraph);

    ggml_build_forward_expand(gf, c);

    // 7. Benchmark and execute
    if (type == GGML_TYPE_F16){
        std::cout << "Starting benchmark FP16 (1024 iterations)..." << std::endl;
    } else {
        std::cout << "Starting benchmark FP32 (1024 iterations)..." << std::endl;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const int iterations = 32;
    for (int i = 0; i < iterations; ++i) {
        for (int j = 0; j < iterations; ++j) {
            ggml_backend_graph_compute(backend, gf);
        }
        std::cout<<".";
    }
    std::cout<<std::endl;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    double avg_time_ms = duration.count() / (iterations * iterations);
    double total_ops = (double)nx * ny * nz;
    double gflops = (total_ops / (avg_time_ms / 1000.0)) / 1000000000.0;

    std::cout << "Average time per addition: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;

    // 9. Cleanup
    ggml_free(ctx);
    ggml_free(ctx_cgraph);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return 0;
}
