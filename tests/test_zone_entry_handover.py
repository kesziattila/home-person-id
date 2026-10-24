"""Zone entry handover (stairs): an unnamed track walking into a zone takes the name of the one
person lost on the same camera just before it appeared.

Live 2026-10-21 18:01 UTC: Ati's named living_room track was lost at the stair foot, a fresh
ByteTrack track climbed 2 s later unnamed (Re-ID 0.56–0.67 from behind) → Ati stayed "living room".
Offline 10-17..21: all 26 climbs were fresh tracks; the 5 s rule gave 7 names, none contradicted
(docs/TUNING_LOG.md 2026-10-21).
"""

from datetime import datetime, timedelta
from pathlib import Path

from src.config import ZoneConfig, ZonePolygon, ZonesConfig, load_config
from src.tracking.track import TrackState
from tests.test_cross_camera_zone_propagation import _make_global_track_manager, _register_active_track

CAM = "living_room"
FLOOR_BOX = (800, 600, 1000, 1000)   # foot y 0.93 -> "living_room"
STAIRS_BOX = (800, 100, 1000, 300)   # foot y 0.28 -> "stairway"


def _manager(handover_sec=5.0, cameras=(CAM,)):
    zones = ZonesConfig(zones=[
        ZoneConfig(name="stairway", cameras={CAM: ZonePolygon([[0, 0], [1, 0], [1, 0.5], [0, 0.5]]),
                                             "ground_floor": ZonePolygon([[0, 0], [1, 0], [1, 1], [0, 1]])},
                   exit_destination="upstairs", entry_handover_sec=handover_sec,
                   entry_handover_cameras=list(cameras)),
        ZoneConfig(name="living_room", cameras={CAM: ZonePolygon([[0, 0.5], [1, 0.5], [1, 1], [0, 1]])}),
    ])
    return _make_global_track_manager(zones_config=zones)


def _lost(manager, local_id, gid, person_id, lost_ago, zone="living_room", camera=CAM):
    """A track on the camera, lost `lost_ago` seconds before now, last seen in `zone`."""
    _register_active_track(manager, camera, local_id, gid, FLOOR_BOX, person_id=person_id,
                           identified_by="reid_gallery" if person_id else "none", confidence=0.78)
    t = manager._tracks[gid]
    t.state = TrackState.GRACE
    t.first_seen = datetime.now() - timedelta(seconds=lost_ago + 30)
    t.last_seen = datetime.now() - timedelta(seconds=lost_ago)
    manager._track_zones[gid] = zone
    # a lost track is no longer evaluated for zones
    del manager._local_to_global[(camera, local_id)]


def _climber(manager, born_ago=1.0, camera=CAM):
    """Unnamed track born at the stair foot `born_ago` s ago, now on the stairs."""
    _register_active_track(manager, camera, 99, "global_climb", STAIRS_BOX)
    manager._tracks["global_climb"].first_seen = datetime.now() - timedelta(seconds=born_ago)
    manager._track_zones["global_climb"] = "living_room"
    manager._update_track_zones()
    return manager.identity_linker.get_track_state("global_climb")


def _handover_events(manager):
    return [c for c in manager.repository.create_event.call_args_list
            if c.kwargs.get("event_type") == "zone_entry_handover"]


def test_climber_gets_name_of_track_lost_at_stair_foot():
    m = _manager()
    _lost(m, 1, "global_ati", person_id=42, lost_ago=3.0)   # lost 2 s before the climber appeared
    state = _climber(m, born_ago=1.0)
    assert state.person_id == 42
    ev = _handover_events(m)
    assert len(ev) == 1 and ev[0].kwargs["extra_data"]["from_track"] == "global_ati"


def test_no_handover_on_camera_not_listed():
    m = _manager(cameras=("ground_floor",))
    _lost(m, 1, "global_ati", person_id=42, lost_ago=3.0)
    assert _climber(m).person_id is None


def test_all_cameras_of_zone_when_list_empty():
    m = _manager(cameras=())
    _lost(m, 1, "global_ati", person_id=42, lost_ago=3.0)
    assert _climber(m).person_id == 42


def test_disabled_by_default():
    m = _manager(handover_sec=0.0)
    _lost(m, 1, "global_ati", person_id=42, lost_ago=3.0)
    assert _climber(m).person_id is None


def test_two_different_people_lost_no_handover():
    m = _manager()
    _lost(m, 1, "global_ati", person_id=42, lost_ago=3.0)
    _lost(m, 2, "global_judit", person_id=7, lost_ago=2.0)
    assert _climber(m).person_id is None


def test_unnamed_lost_track_makes_it_ambiguous():
    m = _manager()
    _lost(m, 1, "global_ati", person_id=42, lost_ago=3.0)
    _lost(m, 2, "global_unknown", person_id=None, lost_ago=2.0)
    assert _climber(m).person_id is None


def test_track_that_went_up_the_stairs_is_not_a_source():
    """A track lost IN the stairway went upstairs; it can't be the next climber."""
    m = _manager()
    _lost(m, 1, "global_ati", person_id=42, lost_ago=3.0, zone="stairway")
    assert _climber(m).person_id is None


def test_lost_too_long_before_birth_no_handover():
    m = _manager(handover_sec=5.0)
    _lost(m, 1, "global_ati", person_id=42, lost_ago=8.0)   # 7 s before the climber appeared
    assert _climber(m, born_ago=1.0).person_id is None


def test_person_held_by_live_track_on_camera_no_handover():
    m = _manager()
    _lost(m, 1, "global_ati_old", person_id=42, lost_ago=3.0)
    _register_active_track(m, CAM, 5, "global_ati_live", FLOOR_BOX, person_id=42,
                           identified_by="face", confidence=0.6)
    assert _climber(m).person_id is None


def test_entered_zone_long_after_birth_no_handover():
    """Live 2026-10-22 07:00 UTC: the successor track entered 90 s after birth, by then on Virag."""
    m = _manager()
    _lost(m, 1, "global_judit", person_id=7, lost_ago=92.0)
    m.zone_manager.get_zone_config("stairway").entry_handover_sec = 5.0
    assert _climber(m, born_ago=90.0).person_id is None


def test_person_live_on_other_camera_outside_zone_no_handover():
    """Live 2026-10-22: Judit was tracked in the work room (ground floor) while Virag climbed."""
    m = _manager()
    _lost(m, 1, "global_judit", person_id=7, lost_ago=3.0)
    _register_active_track(m, "hallway", 3, "global_judit_elsewhere", FLOOR_BOX, person_id=7,
                           identified_by="reid_gallery", confidence=0.8)
    m._track_zones["global_judit_elsewhere"] = "work_room"
    assert _climber(m).person_id is None


def test_person_live_on_overlapping_camera_in_same_zone_still_fires():
    m = _manager()
    _lost(m, 1, "global_ati", person_id=42, lost_ago=3.0)
    _register_active_track(m, "ground_floor", 3, "global_ati_gf", STAIRS_BOX, person_id=42,
                           identified_by="reid_gallery", confidence=0.8)
    m._track_zones["global_ati_gf"] = "stairway"
    assert _climber(m).person_id == 42


def test_track_born_inside_zone_never_fires():
    """Someone appearing on the stairs (coming down) has no zone change into the stairway."""
    m = _manager()
    _lost(m, 1, "global_ati", person_id=42, lost_ago=3.0)
    _register_active_track(m, CAM, 99, "global_down", STAIRS_BOX)
    m._tracks["global_down"].first_seen = datetime.now() - timedelta(seconds=1.0)
    m._track_zones["global_down"] = "stairway"   # initial zone at birth
    m._update_track_zones()
    assert m.identity_linker.get_track_state("global_down").person_id is None


def test_config_parses_entry_handover(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "zones:\n"
        "  - name: stairway\n"
        "    cameras:\n"
        "      living_room:\n"
        "        polygon: [[0, 0], [1, 0], [1, 1]]\n"
        "    entry_handover_sec: 5.0\n"
        "    entry_handover_cameras: [living_room]\n"
    )
    z = load_config(str(p)).zones.zones[0]
    assert z.entry_handover_sec == 5.0 and z.entry_handover_cameras == ["living_room"]


def test_uncertain_track_neither_changes_zone_nor_takes_a_handover():
    """A box that merged with another and has not separated yet (LocalTrack.uncertain) may be
    either person: no zone change, no handover from it until ByteTracker settles it."""
    m = _manager()
    _lost(m, 1, "global_ati", person_id=42, lost_ago=3.0)
    m._uncertain = {(CAM, 99)}
    state = _climber(m)
    assert state.person_id is None and not _handover_events(m)
    assert m._track_zones["global_climb"] == "living_room"       # not moved to "stairway"
