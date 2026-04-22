# ---- GPU verification script

import time
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def check_gpu():
    print("---- Checking GPU availability ----")

    if not CUPY_AVAILABLE:
        print("CuPy not installed. GPU not available.")
        return

    try:
        device = cp.cuda.Device(0)
        print(f"GPU Device: {device}")
    except Exception as e:
        print("Failed to access GPU:", e)
        return


def benchmark():
    print("\n---- Running benchmark (10000x10000 matmul) ----")

    size = 30000

    # ---- CPU
    a_cpu = np.random.rand(size, size).astype(np.float32)
    b_cpu = np.random.rand(size, size).astype(np.float32)

    start = time.time()
    c_cpu = np.dot(a_cpu, b_cpu)
    cpu_time = time.time() - start

    print(f"CPU Time: {cpu_time:.4f} sec")

    if not CUPY_AVAILABLE:
        return

    # ---- GPU
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)

    cp.cuda.Stream.null.synchronize()

    start = time.time()
    c_gpu = cp.dot(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start

    print(f"GPU Time: {gpu_time:.4f} sec")

    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")


def gpu_memory():
    print("\n---- GPU Memory Info ----")

    if not CUPY_AVAILABLE:
        print("CuPy not available")
        return

    mempool = cp.get_default_memory_pool()
    print(f"Used Memory: {mempool.used_bytes() / 1024**2:.2f} MB")
    print(f"Total Memory: {mempool.total_bytes() / 1024**2:.2f} MB")


if __name__ == "__main__":
    check_gpu()
    benchmark()
    gpu_memory()

    print("\n---- RESULT ----")
    print("If GPU time < CPU time → PASS")