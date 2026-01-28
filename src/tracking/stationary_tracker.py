from typing import Optional

class StationaryTracker:
    """Tracks stationary state of tracks."""

    def __init__(self, timeout: float = 0):
        """Initialize stationary tracker.

        Args:
            timeout: Seconds before removing stationary track (0 = no timeout)
        """
        self.timeout = timeout
        self._stationary_since: dict[tuple[str, int], float] = {}  # (camera_id, track_id) -> timestamp

    def update(self, camera_id: str, track_ids: list[int], has_motion: bool, current_time: float):
        """Update stationary state for tracks.

        Args:
            camera_id: Camera identifier
            track_ids: List of current track IDs
            has_motion: Whether motion was detected
            current_time: Current timestamp
        """
        if has_motion:
            # Motion detected - clear stationary timers for these tracks
            for track_id in track_ids:
                self._stationary_since.pop((camera_id, track_id), None)
        else:
            # No motion - mark tracks as stationary
            for track_id in track_ids:
                if (camera_id, track_id) not in self._stationary_since:
                    self._stationary_since[(camera_id, track_id)] = current_time

    def get_stationary_time(self, camera_id: str, track_id: int, current_time: float) -> Optional[int]:
        """Get how long a track has been stationary.

        Returns:
            Seconds stationary, or None if not stationary
        """
        key = (camera_id, track_id)
        if key in self._stationary_since:
            return int(current_time - self._stationary_since[key])
        return None

    def is_stationary(self, camera_id: str, track_id: int) -> bool:
        """Check if a track is stationary."""
        return (camera_id, track_id) in self._stationary_since

    def cleanup_track(self, camera_id: str, track_id: int):
        """Remove stationary tracking for a track."""
        self._stationary_since.pop((camera_id, track_id), None)

    def cleanup_expired(self, current_time: float) -> list[tuple[str, int]]:
        """Remove expired stationary tracks.

        Returns:
            List of (camera_id, track_id) that were removed
        """
        if self.timeout <= 0:
            return []

        expired = []
        for key, start_time in list(self._stationary_since.items()):
            if current_time - start_time > self.timeout:
                expired.append(key)
                del self._stationary_since[key]

        return expired
