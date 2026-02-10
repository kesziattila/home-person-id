import os
import tempfile
from datetime import datetime, timedelta

import numpy as np

from src.config import Config, ReIDConfig, ZonesConfig, CameraTopologyConfig
from src.database.repository import Repository
from src.recognition.identity_linker import IdentityLinker
from src.tracking.global_tracker import GlobalTrackManager
from src.tracking.track import GlobalTrack, TrackState


def make_system(reid_grace_sec: float = 1.0):
    # Minimal config objects
    reid = ReIDConfig(enabled=True, global_id_grace_period=reid_grace_sec)
    cfg = Config(reid=reid)

    # Temp DB path
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    repo = Repository(db_path)

    # IdentityLinker (uses defaults for other configs)
    linker = IdentityLinker(cfg.face_recognition, cfg.reid, repo)

    # Global tracker
    gtm = GlobalTrackManager(
        CameraTopologyConfig(),
        cfg.reid,
        linker,
        repo,
        zones_config=ZonesConfig(),
    )
    return gtm, repo, tmpdir


def test_lost_track_retired_after_grace_period():
    gtm, repo, tmpdir = make_system(0.5)

    # Insert a LOST track with old last_seen
    track_id = "global_1"
    gt = GlobalTrack(track_id=track_id, state=TrackState.LOST)
    gt.last_seen = datetime.now() - timedelta(seconds=5)
    gtm._tracks[track_id] = gt

    # Trigger cleanup logic via the internal method
    now = datetime.now().timestamp()
    gtm._cleanup_lost_tracks(now)

    # The track should be marked REMOVED first
    assert gtm._tracks[track_id].state == TrackState.REMOVED

    # Periodic cleanup should delete REMOVED immediately
    gtm.cleanup_old_tracks(max_age_hours=24)
    assert track_id not in gtm._tracks
