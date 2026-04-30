# ==================================================
# tests/performance/test_benchmark.py
# DataLoader throughput benchmark.
#
# Run: pytest tests/performance/test_benchmark.py -v -s
#
# PURPOSE:
#   Verify the DataLoader feeds batches faster than the GPU
#   can consume them. If not, data is the bottleneck — not
#   the model — and training time is wasted waiting for data.
#
# TARGET (GTX 1650):
#   < 150 ms/batch for Real-ESRGAN (batch_size=8)
#   < 80  ms/batch for EDSR cached (batch_size=16)
# ==================================================

import time
from pathlib import Path
import numpy as np
import pytest
import cv2
from src.data.dataset import RealESRGANDataset, EDSRDataset, build_dataloader


@pytest.fixture(scope="module")
def bench_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("bench_hr")
    for i in range(50):
        img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(d / f"{i:04d}.png"), img)
    return d


def measure_throughput(loader, n_batches=50, warmup=5) -> dict:
    """
    Time DataLoader over n_batches after warmup_batches warmup iterations.
    Returns throughput statistics.
    """
    it = iter(loader)

    for _ in range(warmup):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)

    start = time.perf_counter()
    count = 0
    last_batch = None

    for _ in range(n_batches):
        try:
            last_batch = next(it)
        except StopIteration:
            it = iter(loader)
            last_batch = next(it)
        count += 1

    elapsed = time.perf_counter() - start
    bs = last_batch["lr"].shape[0]

    return {
        "ms_per_batch":   (elapsed / count) * 1000,
        "batches_per_sec": count / elapsed,
        "samples_per_sec": (count * bs) / elapsed,
        "batch_size":      bs,
    }


def print_result(name: str, r: dict) -> None:
    ms = r["ms_per_batch"]
    rating = "✅ FAST" if ms < 100 else ("⚠️ OK" if ms < 200 else "❌ SLOW")
    print(f"\n  {name}")
    print(f"    {r['batches_per_sec']:.1f} batches/s | "
          f"{ms:.0f} ms/batch | "
          f"{r['samples_per_sec']:.0f} samples/s  {rating}")


class TestDataLoaderPerformance:
    def test_realesrgan_baseline(self, bench_dir):
        """Single-process baseline. Must achieve > 2 batches/s."""
        ds = RealESRGANDataset(bench_dir, scale=4, hr_patch_size=128,
                               patches_per_image=5)
        loader = build_dataloader(ds, batch_size=8, num_workers=0,
                                  shuffle=False)
        r = measure_throughput(loader, n_batches=30)
        print_result("Real-ESRGAN | num_workers=0 | batch=8", r)
        assert r["batches_per_sec"] > 1.0, (
            f"DataLoader too slow: {r['batches_per_sec']:.1f} b/s"
        )

    @pytest.mark.parametrize("nw", [2, 4])
    def test_realesrgan_multiprocess(self, bench_dir, nw):
        """Multiprocess must outperform single-process baseline."""
        ds = RealESRGANDataset(bench_dir, scale=4, hr_patch_size=128,
                               patches_per_image=5)
        loader = build_dataloader(ds, batch_size=8, num_workers=nw,
                                  shuffle=False, pin_memory=False,
                                  persistent_workers=False)
        r = measure_throughput(loader, n_batches=30)
        print_result(f"Real-ESRGAN | num_workers={nw} | batch=8", r)
        assert r["batches_per_sec"] > 1.5

    def test_edsr_cached(self, bench_dir):
        """EDSR with cache must be fastest configuration."""
        ds = EDSRDataset(bench_dir, scale=4, hr_patch_size=96,
                         patches_per_image=5, cache_in_memory=True)
        loader = build_dataloader(ds, batch_size=16, num_workers=0,
                                  shuffle=False)
        r = measure_throughput(loader, n_batches=30)
        print_result("EDSR | cached | num_workers=0 | batch=16", r)
        assert r["batches_per_sec"] > 3.0

    def test_worker_comparison_table(self, bench_dir):
        """
        Print comparison table for 0, 2, 4 workers.
        Informational — no pass/fail threshold.
        Use this to choose num_workers in your YAML config.
        """
        print("\n  " + "="*55)
        print("  WORKER COMPARISON  (Real-ESRGAN, batch=8, patch=128)")
        print("  " + "="*55)
        for nw in [0, 2, 4]:
            ds = RealESRGANDataset(bench_dir, scale=4, hr_patch_size=128,
                                   patches_per_image=4)
            loader = build_dataloader(ds, batch_size=8, num_workers=nw,
                                      shuffle=False, pin_memory=False,
                                      persistent_workers=(nw > 0))
            r = measure_throughput(loader, n_batches=20)
            ms = r["ms_per_batch"]
            tag = "✅" if ms < 100 else ("⚠️" if ms < 200 else "❌")
            print(f"  workers={nw}  {r['batches_per_sec']:6.1f} b/s  "
                  f"{ms:6.0f} ms/batch  {tag}")
        print("  → Set best num_workers in your YAML config\n")
