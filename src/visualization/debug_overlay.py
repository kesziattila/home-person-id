"""Debug overlay renderer for Re-ID diagnostics.

Shows real-time debug information on preview frames when
config.debug.reid_overlay is enabled.
"""

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.recognition.identity_linker import IdentityLinker, TrackIdentityState


class DebugOverlayRenderer:
    """Renders debug information on preview frames.

    Panels:
    - Top-left: Gallery state (person names + embedding counts), pending handovers
    - Per-track: Re-ID gallery size, identification method + age
    - Bottom-right: Last N Re-ID events as a scrolling ticker
    """

    def __init__(self, identity_linker: IdentityLinker, max_events: int = 5):
        self._identity_linker = identity_linker
        self._recent_events: deque[str] = deque(maxlen=max_events)

    def add_event(self, text: str):
        """Add a Re-ID event to the ticker."""
        timestamp = time.strftime("%H:%M:%S")
        self._recent_events.appendleft(f"[{timestamp}] {text}")

    def draw(
        self,
        frame: np.ndarray,
        camera_id: str,
        track_states: Optional[dict[str, TrackIdentityState]] = None,
    ):
        """Draw debug overlay on frame.

        Args:
            frame: Frame to draw on (modified in place)
            camera_id: Camera identifier
            track_states: Track states visible on this camera (if available)
        """
        self._draw_gallery_panel(frame)
        self._draw_event_ticker(frame)

    def _draw_gallery_panel(self, frame: np.ndarray):
        """Draw gallery state in top-left corner."""
        gallery_mgr = self._identity_linker.reid_gallery_manager
        if not gallery_mgr:
            return

        lines = [f"Re-ID Gallery ({gallery_mgr.gallery_size} persons):"]
        for name in gallery_mgr.gallery_persons[:8]:
            entry = gallery_mgr._gallery.get(name)
            if entry:
                age = int(time.time() - entry.last_seen)
                lines.append(f"  {name}: {len(entry.entries)} emb, {age}s ago")

        # Draw semi-transparent background
        line_h = 18
        padding = 6
        panel_h = len(lines) * line_h + padding * 2
        panel_w = 300
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(lines):
            y = padding + (i + 1) * line_h
            cv2.putText(frame, line, (padding, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    def _draw_event_ticker(self, frame: np.ndarray):
        """Draw recent Re-ID events in bottom-right corner."""
        if not self._recent_events:
            return

        h, w = frame.shape[:2]
        line_h = 18
        padding = 6
        lines = list(self._recent_events)
        panel_h = len(lines) * line_h + padding * 2
        panel_w = 420
        x_start = w - panel_w

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(lines):
            y = h - panel_h + padding + (i + 1) * line_h
            cv2.putText(frame, line, (x_start + padding, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
