"""VLM prompts, JSON parsing, and result dataclasses for scene understanding."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.config import VLMConfig
from src.vlm.vlm_client import VLMClient

logger = logging.getLogger(__name__)

# Prompt for per-person activity/appearance analysis
_PERSON_PROMPT_SINGLE = (
    'Analyze this person. Reply with JSON only:\n'
    '{"activity":"<sitting|standing|walking|at_door|lying_down|other>",'
    '"appearance":"<clothing in 5 words or null>",'
    '"gender":"<male|female|unknown>",'
    '"age_group":"<child|teen|adult|elderly|unknown>"}'
)

_PERSON_PROMPT_MULTI = (
    'Here are multiple views of the same person from different cameras. '
    'Analyze the person. Reply with JSON only:\n'
    '{"activity":"<sitting|standing|walking|at_door|lying_down|other>",'
    '"appearance":"<clothing in 5 words or null>",'
    '"gender":"<male|female|unknown>",'
    '"age_group":"<child|teen|adult|elderly|unknown>"}'
)

_VALID_ACTIVITIES = {"sitting", "standing", "walking", "at_door", "lying_down", "other"}
_VALID_GENDERS = {"male", "female", "unknown"}
_VALID_AGE_GROUPS = {"child", "teen", "adult", "elderly", "unknown"}


@dataclass
class VLMResult:
    """Result of per-person VLM analysis."""

    activity: Optional[str] = None
    appearance: Optional[str] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "activity": self.activity,
            "appearance": self.appearance,
            "gender": self.gender,
            "age_group": self.age_group,
        }


@dataclass
class HouseOverview:
    """Aggregated house overview from VLM."""

    summary: str = ""
    persons: list[dict] = field(default_factory=list)
    camera_scenes: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "persons": self.persons,
            "camera_scenes": self.camera_scenes,
            "timestamp": self.timestamp,
        }


class VLMAnalyzer:
    """Runs VLM calls for person analysis and house overview.

    Performance: ~200-2000ms per call depending on model and hardware.
    Calls are made in background threads to avoid blocking the main loop.
    """

    def __init__(self, config: VLMConfig):
        self._config = config
        self._client = VLMClient(config)

    def analyze_person(self, crops: list[np.ndarray]) -> Optional[VLMResult]:
        """Analyze a person using one or more crops from different cameras.

        Performance: ~200-2000ms per call (network + model inference).

        Args:
            crops: List of person crops (BGR numpy arrays). Multiple crops
                   improve accuracy when the same person is seen from different angles.

        Returns:
            VLMResult with activity, appearance, gender, age_group, or None on failure.
        """
        if not crops:
            return None

        prompt = _PERSON_PROMPT_MULTI if len(crops) > 1 else _PERSON_PROMPT_SINGLE

        from src.utils.profiler import profiler
        with profiler.measure("VLM.analyze_person"):
            raw = self._client.call(prompt, images=crops)

        if not raw:
            return None

        return self._parse_person_result(raw)

    def generate_house_overview(
        self,
        person_states: list[dict],
        frames: dict[str, np.ndarray],
        camera_names: Optional[dict[str, str]] = None,
    ) -> Optional[HouseOverview]:
        """Generate house overview in a single VLM call with all camera frames.

        Sends all frames as images with a structured prompt. The model returns JSON
        with a one-sentence summary and per-camera scene descriptions.

        Performance: ~500-3000ms per call depending on number of cameras and hardware.

        Args:
            person_states: List of dicts with name, zone, activity, gender, age_group
            frames: Dict of camera_id -> BGR numpy array
            camera_names: Optional dict of camera_id -> human-readable name

        Returns:
            HouseOverview with summary and camera_scenes populated, or None on failure.
        """
        if not frames and not person_states:
            return None

        camera_names = camera_names or {}
        ordered_ids = list(frames.keys())
        ordered_frames = [frames[cam_id] for cam_id in ordered_ids]

        prompt = self._build_overview_prompt(person_states, ordered_ids, camera_names)

        image_max_size = self._config.overview_image_size if self._config.overview_image_size > 0 else None

        from src.utils.profiler import profiler
        with profiler.measure("VLM.house_overview"):
            raw = self._client.call(
                prompt,
                images=ordered_frames if ordered_frames else None,
                max_tokens=self._config.overview_max_tokens,
                image_max_size=image_max_size,
            )

        if not raw:
            return None

        # Parse JSON response
        summary = ""
        camera_scenes: dict[str, str] = {}
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            summary = str(data.get("summary", "")).strip()
            scenes_raw = data.get("scenes", {})
            if isinstance(scenes_raw, dict):
                camera_scenes = {k: str(v).strip() for k, v in scenes_raw.items() if v}
        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"VLM overview parse error: {e} | raw={raw!r}")
            # Use raw text as summary if JSON parse fails
            summary = raw.strip()

        if not summary:
            return None

        return HouseOverview(
            summary=summary,
            persons=person_states,
            camera_scenes=camera_scenes,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def _parse_person_result(self, raw: str) -> Optional[VLMResult]:
        """Parse JSON response from person analysis prompt."""
        # Extract JSON from response (model may include surrounding text)
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"VLM person result parse error: {e} | raw={raw!r}")
            return None

        activity = data.get("activity")
        if activity not in _VALID_ACTIVITIES:
            activity = "other"

        gender = data.get("gender")
        if gender not in _VALID_GENDERS:
            gender = "unknown"

        age_group = data.get("age_group")
        if age_group not in _VALID_AGE_GROUPS:
            age_group = "unknown"

        appearance = data.get("appearance")
        if not isinstance(appearance, str) or not appearance.strip():
            appearance = None

        return VLMResult(
            activity=activity,
            appearance=appearance,
            gender=gender,
            age_group=age_group,
        )

    def _build_overview_prompt(
        self,
        person_states: list[dict],
        camera_ids: list[str],
        camera_names: dict[str, str],
    ) -> str:
        """Build a compact prompt for the single-call house overview."""
        lines = ["Home cameras:"]
        for i, cam_id in enumerate(camera_ids, 1):
            display = camera_names.get(cam_id, cam_id)
            lines.append(f"{i}. {cam_id} ({display})")

        if person_states:
            parts = []
            for p in person_states:
                name = p.get("name", "Unknown")
                zone = p.get("zone") or "unknown location"
                activity = p.get("activity") or ""
                gender = p.get("gender") or ""
                age = p.get("age_group") or ""
                detail = ", ".join(x for x in [activity, gender, age] if x and x != "unknown")
                parts.append(f"{name} in {zone}" + (f" ({detail})" if detail else ""))
            lines.append("People detected: " + ". ".join(parts) + ".")
        else:
            lines.append("No people detected.")

        cam_keys = ", ".join(f'"{c}":"<one sentence>"' for c in camera_ids)
        lines.append(
            'Reply with JSON only:\n'
            '{"summary":"<one sentence>",'
            f'"scenes":{{{cam_keys}}}}}'
        )
        return "\n".join(lines)
