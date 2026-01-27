import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
import threading

logger = logging.getLogger(__name__)

@dataclass
class ModuleStats:
    name: str
    total_time: float = 0.0
    call_count: int = 0
    max_time: float = 0.0
    
    def reset(self):
        self.total_time = 0.0
        self.call_count = 0
        self.max_time = 0.0

class Profiler:
    def __init__(self, log_interval: float = 10.0):
        self.log_interval = log_interval
        self.stats: Dict[str, ModuleStats] = {}
        self.last_log_time = time.time()
        self._lock = threading.Lock()
        self.enabled = False

    def start_measure(self) -> float:
        return time.perf_counter()

    def end_measure(self, module_name: str, start_time: float):
        elapsed = time.perf_counter() - start_time
        with self._lock:
            if module_name not in self.stats:
                self.stats[module_name] = ModuleStats(name=module_name)
            s = self.stats[module_name]
            s.total_time += elapsed
            s.call_count += 1
            if elapsed > s.max_time:
                s.max_time = elapsed

    def measure(self, module_name: str):
        """Context manager for measuring time."""
        class MeasureContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start_time = None

            def __enter__(self):
                self.start_time = self.profiler.start_measure()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.profiler.end_measure(self.name, self.start_time)

        return MeasureContext(self, module_name)

    def log_stats(self, extra_info: Optional[Dict[str, Any]] = None):
        now = time.time()
        if now - self.last_log_time < self.log_interval:
            return False

        if not self.enabled:
            # Just reset stats if disabled so they don't accumulate forever
            with self._lock:
                for s in self.stats.values():
                    s.reset()
                self.last_log_time = now
            return False

        with self._lock:
            if not self.stats and not extra_info:
                self.last_log_time = now
                return False

            print(f"\n--- Performance Report (last {now - self.last_log_time:.1f}s) ---")
            
            if self.stats:
                # Sort by total time descending
                sorted_stats = sorted(self.stats.values(), key=lambda x: x.total_time, reverse=True)
                
                print(f"{'Module':<30} | {'Calls':<8} | {'Avg (ms)':<10} | {'Max (ms)':<10} | {'Total (s)':<10}")
                print("-" * 75)
                
                for s in sorted_stats:
                    avg = (s.total_time / s.call_count * 1000) if s.call_count > 0 else 0
                    max_ms = s.max_time * 1000
                    print(f"{s.name:<30} | {s.call_count:<8} | {avg:<10.2f} | {max_ms:<10.2f} | {s.total_time:<10.3f}")
            
            if extra_info:
                if self.stats:
                    print("-" * 75)
                for k, v in extra_info.items():
                    print(f"{k:<30}: {v}")
            
            print("-----------------------------------------------------------\n")
            
            # Reset stats
            for s in self.stats.values():
                s.reset()
            self.last_log_time = now
            return True

# Global profiler instance
profiler = Profiler()
