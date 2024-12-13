
# - Double-precision 


When implementing the CUDA-accelerated SVR algorithm, we encountered significant performance bottlenecks during training. Further investigation revealed that this was primarily due to the architectural limitations of the RTX 2060 GPU in handling double-precision floating-point operations.
The RTX 2060's architectural specifications show a pronounced disparity between single and double-precision computing capabilities:
- Single-precision (FP32) performance: 6.5 TFLOPS
- Double-precision (FP64) performance: 203.1 GFLOPS
- **FP64:FP32 ratio: 1:32**

This architectural design significantly impacts the performance of double-precision computations in several ways:

1. **Computational Throughput**
    - Double-precision operations take approximately 32 times longer to process compared to single-precision operations
    - This creates a substantial computational bottleneck in algorithms requiring extensive floating-point calculations
2. **Memory Bandwidth Utilization**
    - Double-precision values require 8 bytes of storage compared to 4 bytes for single-precision
    - This effectively halves the memory bandwidth efficiency
    - Results in increased memory transfer times between host and device
3. **Cache Efficiency**
    - The larger size of double-precision values reduces cache hit rates
    - Leads to more frequent cache misses and memory access operations
    - Further compounds the performance impact

### Impact on SVR Implementation

While double-precision calculations are crucial for maintaining numerical accuracy in financial computations, this architectural limitation presents a significant trade-off between precision and performance. The observed extended training times are therefore an expected consequence of the hardware's architectural design rather than an implementation inefficiency.

