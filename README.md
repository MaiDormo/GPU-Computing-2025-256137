# GPU-Computing-2025-256137

This repository contains various implementations of Sparse Matrix-Vector Multiplication (SpMV) for both CPU and GPU, developed as part of the GPU Computing graduate course at the University of Trento.

## Repository structure
```text
├── bin/                    # Compiled executables
├── data/                   # Matrix market test files
├── include/                # Header files
├── lib/                    # Library implementation files
├── obj/                    # Object files
├── src/                    # Source code for all implementations
│   ├── spmv_cpu_*.c        # CPU implementations (simple, omp, ilp)
│   └── spmv_gpu_*.cu       # GPU implementations (simple, value_sequential, strided,vector)
├── deviceQuery/            # NVIDIA device information utilities
├── *.sh                    # Benchmark and run scripts
└── extract_spmv_data.sh    # Script to extract and process benchmark results
```

Inside the code there is also the "test" folder, inside this folder lies most of the code developed during the lab lesson, so feel free to skip it.

## Implementations

CPU Implementations

- **Simple:** Basic CSR implementation (one thread per row)
- **ILP:** Instruction-level parallelism optimization

GPU Implementations

- **Simple:** Basic CSR implementation (one thread per row)
- **Value Sequential:** Thread per non-zero element
- **Strided:** Value-parallel with strided access pattern

## How to Compile

To compile all implementations:
```bash
make
```
To clean up build artifacts:
```bash
make clean
```

Keep in mind that if this code is runned on the L40S gpu the makefile line:

```makefile
NV_OPT := --gpu-architecture=sm_80 -m64 -Xcompiler -fopenmp
```

needs to be changed like this:
```makefile
NV_OPT := --gpu-architecture=sm_89 -m64 -Xcompiler -fopenmp
```
(if already compiled with the older architecture, you need to clean and rebuild).




## How to Download the Sparse Matrices

In order to download the dataset, you can run the bash script called download_matrices.sh. This script is going to take care of the download and unpacking of the matrices, positioning them inside the data folder.

If some error occurs during the download, the links are inside the script and you can download them independently following the command inside it.

## How to Run

Single test
```bash
# CPU implementation
./bin/spmv_cpu_<impl> data/<dataset>/<dataset>.mtx

# GPU implementation
./bin/spmv_gpu_<impl>.exec data/<dataset>/<dataset>.mtx
```

Batch Benchmarking
```bash
# Run all benchmarks
./run_all_benchmarks.sh

# Run specific implementation
./cpu_simple_run.sh
./gpu_simple_run.sh
```
Data analysis

```bash
./extract_spmv_data.sh
```

This will generate spmv_benchmark_results.csv with performance metrics for all implementations and datasets.


In order to run the experiment with different kernels you need to use a different script

```bash
sbatch run_spmv_experimenting_block_thread_sizes.sh
```

## Performance Metrics

The benchmarks measure:
- Execution time (s)
- Memory Bandwidth (GB/s)
- Computational Perfromance (GFLOPS)

## Hardware Used

- **CPU**: intel something, AMD EPYC (something)
- **GPU**: NVIDIA RTX A30, NVIDIA RTX L40S
- Cuda Toolkit 12.3.2 or later

