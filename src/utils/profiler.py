import time
import logging
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple
import threading

logger = logging.getLogger(__name__)


# ==================== Memory Profiler ====================

@dataclass
class MemorySnapshot:
    """A memory snapshot for comparison."""
    name: str
    timestamp: float
    python_heap_mb: float
    cuda_allocated_mb: float
    cuda_reserved_mb: float
    process_rss_mb: float = 0.0  # Actual process memory (RSS)
    tracemalloc_snapshot: Optional[Any] = None  # tracemalloc.Snapshot


@dataclass
class DataStructureSize:
    """Size tracking for a specific data structure."""
    name: str
    count: int
    size_bytes: int = 0


class MemoryProfiler:
    """Memory profiler for tracking Python heap and CUDA memory.

    Usage:
        from src.utils.profiler import memory_profiler

        # After model initialization:
        memory_profiler.take_snapshot("after_init")

        # Periodically during processing:
        memory_profiler.take_snapshot("frame_1000")

        # Get leak report:
        memory_profiler.print_leak_report("after_init", "frame_1000")

        # Or use auto-tracking:
        memory_profiler.start_tracking()  # Takes baseline
        # ... process frames ...
        memory_profiler.check_for_leaks()  # Compares to baseline
    """

    def __init__(self):
        self._snapshots: Dict[str, MemorySnapshot] = {}
        self._baseline_name: Optional[str] = None
        self._tracking_enabled = False
        self._lock = threading.Lock()
        self._data_structure_callbacks: List[callable] = []
        self._check_interval = 60.0  # seconds
        self._last_check_time = 0.0

    def enable_tracemalloc(self):
        """Enable tracemalloc for detailed Python memory tracking."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Keep 25 frames for traceback
            logger.info("tracemalloc enabled")

    def take_snapshot(self, name: str) -> MemorySnapshot:
        """Take a memory snapshot.

        Args:
            name: Unique name for this snapshot (e.g., "after_init", "frame_1000")

        Returns:
            MemorySnapshot with current memory state
        """
        python_heap_mb = 0.0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            python_heap_mb = current / (1024 * 1024)

        # Get process RSS (actual memory usage) - works on Linux/Mac
        process_rss_mb = 0.0
        try:
            import resource
            # ru_maxrss is in KB on Linux, bytes on Mac
            import platform
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if platform.system() == 'Darwin':
                process_rss_mb = usage.ru_maxrss / (1024 * 1024)  # bytes -> MB
            else:
                process_rss_mb = usage.ru_maxrss / 1024  # KB -> MB
        except ImportError:
            pass

        # Get CUDA memory if available
        cuda_allocated_mb = 0.0
        cuda_reserved_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                cuda_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                cuda_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        except ImportError:
            pass

        # Take tracemalloc snapshot for detailed comparison
        tm_snapshot = None
        if tracemalloc.is_tracing():
            tm_snapshot = tracemalloc.take_snapshot()

        snapshot = MemorySnapshot(
            name=name,
            timestamp=time.time(),
            python_heap_mb=python_heap_mb,
            cuda_allocated_mb=cuda_allocated_mb,
            cuda_reserved_mb=cuda_reserved_mb,
            process_rss_mb=process_rss_mb,
            tracemalloc_snapshot=tm_snapshot,
        )

        with self._lock:
            self._snapshots[name] = snapshot

        return snapshot

    def start_tracking(self, baseline_name: str = "baseline"):
        """Start memory tracking with a baseline snapshot.

        Call this after model initialization.
        """
        self.enable_tracemalloc()
        self.take_snapshot(baseline_name)
        self._baseline_name = baseline_name
        self._tracking_enabled = True
        self._last_check_time = time.time()
        logger.info(f"Memory tracking started with baseline '{baseline_name}'")

    def register_data_structure(self, callback: callable):
        """Register a callback that returns DataStructureSize list.

        The callback should return a list of DataStructureSize objects
        representing important data structures to track.

        Example:
            def get_tracker_sizes():
                return [
                    DataStructureSize("_tracks", len(tracker._tracks)),
                    DataStructureSize("_track_states", len(linker._track_states)),
                ]
            memory_profiler.register_data_structure(get_tracker_sizes)
        """
        self._data_structure_callbacks.append(callback)

    def get_data_structure_sizes(self) -> List[DataStructureSize]:
        """Get sizes of all registered data structures."""
        sizes = []
        for callback in self._data_structure_callbacks:
            try:
                sizes.extend(callback())
            except Exception as e:
                logger.warning(f"Error getting data structure size: {e}")
        return sizes

    def check_for_leaks(self, force: bool = False) -> Optional[str]:
        """Check for memory leaks compared to baseline.

        Args:
            force: Force check even if interval hasn't elapsed

        Returns:
            Leak report string if leaks detected, None otherwise
        """
        if not self._tracking_enabled or not self._baseline_name:
            return None

        current_time = time.time()
        if not force and (current_time - self._last_check_time) < self._check_interval:
            return None

        self._last_check_time = current_time

        # Take current snapshot
        current_name = f"check_{int(current_time)}"
        self.take_snapshot(current_name)

        report = self.get_leak_report(self._baseline_name, current_name)

        # Clean up check snapshot to avoid memory growth from profiler itself
        with self._lock:
            if current_name in self._snapshots:
                del self._snapshots[current_name]

        return report

    def get_leak_report(self, baseline_name: str, current_name: str) -> str:
        """Compare two snapshots and report potential leaks.

        Args:
            baseline_name: Name of baseline snapshot
            current_name: Name of current snapshot

        Returns:
            Formatted report string
        """
        with self._lock:
            baseline = self._snapshots.get(baseline_name)
            current = self._snapshots.get(current_name)

        if not baseline or not current:
            return "Missing snapshots for comparison"

        lines = []
        elapsed = current.timestamp - baseline.timestamp
        lines.append(f"\n=== Memory Report ({elapsed:.1f}s since baseline) ===")

        # Overall memory
        python_diff = current.python_heap_mb - baseline.python_heap_mb
        cuda_diff = current.cuda_allocated_mb - baseline.cuda_allocated_mb
        rss_diff = current.process_rss_mb - baseline.process_rss_mb

        lines.append(f"\nProcess RSS:  {current.process_rss_mb:.1f} MB ({rss_diff:+.1f} MB)  <- actual memory usage")
        lines.append(f"Python heap:  {current.python_heap_mb:.1f} MB ({python_diff:+.1f} MB)  <- tracked allocations")
        lines.append(f"CUDA alloc:   {current.cuda_allocated_mb:.1f} MB ({cuda_diff:+.1f} MB)")
        lines.append(f"CUDA reserved: {current.cuda_reserved_mb:.1f} MB")

        # Data structure sizes
        ds_sizes = self.get_data_structure_sizes()
        if ds_sizes:
            lines.append(f"\nData Structures:")
            for ds in ds_sizes:
                lines.append(f"  {ds.name}: {ds.count} entries")

        # Top memory allocations (if tracemalloc available)
        if baseline.tracemalloc_snapshot and current.tracemalloc_snapshot:
            lines.append(f"\nTop Memory Growth (by location):")
            top_stats = current.tracemalloc_snapshot.compare_to(
                baseline.tracemalloc_snapshot, 'lineno'
            )
            for stat in top_stats[:10]:
                if stat.size_diff > 1024:  # Only show if > 1KB growth
                    size_kb = stat.size_diff / 1024
                    lines.append(f"  {stat.traceback.format()[0].strip()}: +{size_kb:.1f} KB")

        lines.append("=" * 50 + "\n")

        return "\n".join(lines)

    def print_current_memory(self):
        """Print current memory usage."""
        snapshot = self.take_snapshot("_current")
        print(f"\n--- Current Memory ---")
        print(f"Python heap:   {snapshot.python_heap_mb:.1f} MB")
        print(f"CUDA allocated: {snapshot.cuda_allocated_mb:.1f} MB")
        print(f"CUDA reserved:  {snapshot.cuda_reserved_mb:.1f} MB")

        ds_sizes = self.get_data_structure_sizes()
        if ds_sizes:
            print(f"\nData Structures:")
            for ds in ds_sizes:
                print(f"  {ds.name}: {ds.count} entries")
        print("-" * 25 + "\n")

        # Clean up
        with self._lock:
            if "_current" in self._snapshots:
                del self._snapshots["_current"]

    def get_init_memory_report(self) -> str:
        """Get report of memory used by initialization.

        Call this after model loading to see CUDA memory footprint.
        """
        lines = ["\n=== Initialization Memory Footprint ==="]

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                lines.append(f"CUDA allocated: {allocated:.1f} MB")
                lines.append(f"CUDA reserved:  {reserved:.1f} MB")

                # Get per-device info if multiple GPUs
                for i in range(torch.cuda.device_count()):
                    alloc = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    lines.append(f"  GPU {i}: {alloc:.1f} MB")
        except ImportError:
            lines.append("PyTorch not available")

        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            lines.append(f"Python heap: {current / (1024*1024):.1f} MB (peak: {peak / (1024*1024):.1f} MB)")

        lines.append("=" * 40 + "\n")
        return "\n".join(lines)


# Global memory profiler instance
memory_profiler = MemoryProfiler()

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


def create_data_structure_tracker(
    global_tracker=None,
    identity_linker=None,
    id_manager=None,
) -> callable:
    """Create a callback for tracking data structure sizes.

    Pass in references to your main objects and register with memory_profiler.

    Example:
        from src.utils.profiler import memory_profiler, create_data_structure_tracker

        tracker_callback = create_data_structure_tracker(
            global_tracker=self.global_tracker,
            identity_linker=self.identity_linker,
            id_manager=self.id_manager,
        )
        memory_profiler.register_data_structure(tracker_callback)
    """
    def get_sizes() -> List[DataStructureSize]:
        sizes = []

        if global_tracker:
            sizes.append(DataStructureSize("GlobalTracker._tracks", len(global_tracker._tracks)))
            sizes.append(DataStructureSize("GlobalTracker._local_to_global", len(global_tracker._local_to_global)))
            sizes.append(DataStructureSize("GlobalTracker._recently_lost", len(global_tracker._recently_lost_tracks)))
            sizes.append(DataStructureSize("GlobalTracker._pending_handovers", len(global_tracker._pending_handovers)))

        if identity_linker:
            sizes.append(DataStructureSize("IdentityLinker._track_states", len(identity_linker._track_states)))

        if id_manager:
            sizes.append(DataStructureSize("IdentificationManager._identities", len(id_manager._identities)))
            if hasattr(id_manager, '_face_gallery') and id_manager._face_gallery:
                sizes.append(DataStructureSize("IdentificationManager._face_gallery", len(id_manager._face_gallery)))

        return sizes

    return get_sizes
