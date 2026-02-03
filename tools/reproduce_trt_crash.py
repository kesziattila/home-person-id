#!/usr/bin/env python3
"""
Reproduction script for TensorRT thread-safety issues.
This script stressors the TensorRT engine by running concurrent inferences from multiple threads.

Usage:
    # Test with current (fixed) implementation:
    PYTHONPATH=. python tools/reproduce_trt_crash.py --threads 4 --iterations 50

    # Attempt to reproduce the crash (BYPASSING THE LOCK):
    # WARNING: This will likely freeze the GPU/System on Jetson!
    PYTHONPATH=. python tools/reproduce_trt_crash.py --threads 4 --iterations 50 --reproduce-crash
"""

import argparse
import threading
import time
import sys
import logging
from pathlib import Path
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ReproductionTest")

sys.path.insert(0, str(Path(__file__).parent.parent))

def monkeypatch_disable_lock():
    """Monkeypatch TensorRTEngine to disable the synchronization lock."""
    from src.inference.tensorrt_base import TensorRTEngine
    
    logger.warning("!!! MONKEYPATCHING: DISABLING THREAD-SAFETY LOCK !!!")
    
    # Store original method
    original_run = TensorRTEngine._run_inference
    
    def broken_run_inference(self, input_tensor):
        """Version of _run_inference that ignores self._lock."""
        self._load_engine()
        # skip: with self._lock:
        self._cuda_context.push()
        try:
            input_host, input_device = self._get_input_buffer()
            np.copyto(input_host, input_tensor)
            self._cuda.memcpy_htod_async(input_device, input_host, self._stream)
            if not self._context.execute_async_v3(stream_handle=self._stream.handle):
                return []
            outputs = []
            for name in self._output_names:
                host, device, _ = self._buffers[name]
                self._cuda.memcpy_dtoh_async(host, device, self._stream)
                outputs.append(host)
            self._stream.synchronize()
            return [out.copy() for out in outputs]
        except Exception as e:
            logger.error(f"Inference failed (expected if crashing): {e}")
            return []
        finally:
            self._cuda_context.pop()
            
    TensorRTEngine._run_inference = broken_run_inference

def worker_thread(embedder, thread_id, iterations, results):
    """Worker thread that runs inference repeatedly."""
    logger.info(f"Thread-{thread_id} started")
    dummy_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    
    success_count = 0
    for i in range(iterations):
        try:
            # We use extract which calls _run_inference
            emb = embedder.extract(dummy_crop)
            if emb is not None and len(emb) > 0:
                success_count += 1
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Thread-{thread_id}: Completed {i+1}/{iterations}")
        except Exception as e:
            logger.error(f"Thread-{thread_id} error at iter {i}: {e}")
            break
            
    results[thread_id] = success_count
    logger.info(f"Thread-{thread_id} finished ({success_count}/{iterations} successful)")

def main():
    parser = argparse.ArgumentParser(description="TensorRT Thread-Safety Stress Test")
    parser.add_argument("--model", type=str, default="models/osnet_ain_x1_0.engine", help="Path to Re-ID engine")
    parser.add_argument("--threads", type=int, default=4, help="Number of concurrent threads")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per thread")
    parser.add_argument("--reproduce-crash", action="store_true", help="Disable the lock to reproduce the crash")
    args = parser.parse_args()

    if args.reproduce_crash:
        monkeypatch_disable_lock()

    from src.recognition.tensorrt_reid import TensorRTReIDEmbedder
    
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return

    logger.info(f"Initializing embedder with model: {args.model}")
    embedder = TensorRTReIDEmbedder(args.model)
    
    # Warmup
    logger.info("Warming up...")
    embedder.warmup()

    threads = []
    results = [0] * args.threads
    
    start_time = time.time()
    logger.info(f"Starting {args.threads} threads, {args.iterations} iterations each...")
    
    for i in range(args.threads):
        t = threading.Thread(target=worker_thread, args=(embedder, i, args.iterations, results))
        threads.append(t)
        t.start()

    # Monitor threads
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")

    for t in threads:
        t.join(timeout=1.0)

    duration = time.time() - start_time
    total_success = sum(results)
    total_iters = args.threads * args.iterations
    
    logger.info("=" * 40)
    logger.info(f"Test Finished in {duration:.2f}s")
    logger.info(f"Total Success: {total_success}/{total_iters}")
    logger.info(f"Throughput: {total_success/duration:.2f} inf/sec")
    
    if total_success == total_iters:
        logger.info("RESULT: PASS (All inferences succeeded)")
    else:
        logger.error(f"RESULT: FAIL ({total_iters - total_success} inferences failed or hung)")

if __name__ == "__main__":
    main()
