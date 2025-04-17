# NVIDIA L40S GPU Specifications

## Basic Information
| Specification | Value |
|---------------|-------|
| Device | NVIDIA L40S |
| CUDA Driver / Runtime Version | 12.4 / 12.4 |
| CUDA Capability Major/Minor | 8.9 |
| Total Global Memory | 45589 MBytes (47803596800 bytes) |

## Core Architecture
| Specification | Value |
|---------------|-------|
| Multiprocessors | 142 |
| CUDA Cores per MP | 128 |
| Total CUDA Cores | 18176 |
| GPU Max Clock Rate | 2520 MHz (2.52 GHz) |

## Memory Specifications
| Specification | Value |
|---------------|-------|
| Memory Clock Rate | 9001 Mhz |
| Memory Bus Width | 384-bit |
| L2 Cache Size | 100663296 bytes |
| Constant Memory | 65536 bytes |
| Shared Memory per Block | 49152 bytes |
| Shared Memory per Multiprocessor | 102400 bytes |

## Thread & Block Limits
| Specification | Value |
|---------------|-------|
| Warp Size | 32 |
| Max Threads per Multiprocessor | 1536 |
| Max Threads per Block | 1024 |
| Max Thread Block Dimensions | (1024, 1024, 64) |
| Max Grid Dimensions | (2147483647, 65535, 65535) |

## Texture & Memory Features
| Specification | Value |
|---------------|-------|
| Max Texture Dimensions 1D | 131072 |
| Max Texture Dimensions 2D | (131072, 65536) |
| Max Texture Dimensions 3D | (16384, 16384, 16384) |
| Max Layered 1D Texture | 1D=(32768), 2048 layers |
| Max Layered 2D Texture | 2D=(32768, 32768), 2048 layers |
| Maximum Memory Pitch | 2147483647 bytes |
| Texture Alignment | 512 bytes |
| Registers Available per Block | 65536 |

## Advanced Features
| Feature | Support |
|---------|---------|
| Concurrent Copy & Kernel Execution | Yes with 2 copy engine(s) |
| Run time limit on kernels | No |
| Integrated GPU sharing Host Memory | No |
| Support host page-locked memory mapping | Yes |
| Alignment requirement for Surfaces | Yes |
| ECC support | Enabled |
| Unified Addressing (UVA) | Yes |
| Managed Memory | Yes |
| Compute Preemption | Yes |
| Cooperative Kernel Launch | Yes |
| MultiDevice Co-op Kernel Launch | Yes |
| Device PCI Domain/Bus/location ID | 0 / 161 / 0 |
| Compute Mode | Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) |