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


# Field descriptions are the single source of truth for the per-field rules:
# llama.cpp's grammar converter IGNORES schema descriptions (the model never
# sees the schema), so _build_overview_instruction() renders them into the
# prompt text. Edit them here, both schema and prompt follow.
_SCENE_FIELD_DESC = (
    "one sentence describing only what is visible in this camera's image; "
    "mention people only if they are visible in it"
)
_SUMMARY_FIELD_DESC = (
    "one sentence summarizing the whole house, combining the visible scenes "
    "with the sensor data"
)


def _build_overview_schema(camera_ids: list[str]) -> dict:
    """Build the JSON schema for the house overview response.

    Each camera id becomes a required key of "scenes", so the model cannot
    invent or drop cameras. The schema is grammar-enforced by the VLM server
    during sampling (see VLMClient._json_schema_kwargs), guaranteeing valid
    raw JSON — no markdown fences, no missing keys.

    Property order matters: "scenes" comes FIRST so the grammar forces the
    model to describe each camera image before writing the summary. With
    summary-first, the person-state context dominated generation and the
    model filled every scene by repeating its own summary (live incident
    2026-06-07: all four cameras said "A person is in the work room.").
    """
    return {
        "type": "object",
        "properties": {
            "scenes": {
                "type": "object",
                "description": "per-camera scene descriptions",
                "properties": {
                    cam_id: {"type": "string", "description": _SCENE_FIELD_DESC}
                    for cam_id in camera_ids
                },
                "required": list(camera_ids),
                "additionalProperties": False,
            },
            "summary": {"type": "string", "description": _SUMMARY_FIELD_DESC},
        },
        "required": ["scenes", "summary"],
        "additionalProperties": False,
    }

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
        image_max_size = self._config.overview_image_size if self._config.overview_image_size > 0 else None

        # Person context goes FIRST, far from the generation point: when it sat
        # directly before the JSON instruction, the model echoed it as the
        # summary and (grammar forcing all scene keys) repeated that sentence
        # for every camera, ignoring the images (live incident 2026-06-07).
        # Interleave camera label + image so the model can associate each image
        # with its camera name. Sending all images first then the prompt causes
        # the model to describe all cameras identically.
        schema = _build_overview_schema(ordered_ids)
        parts: list = [self._build_overview_context(person_states)]
        for i, cam_id in enumerate(ordered_ids):
            display = camera_names.get(cam_id, cam_id)
            parts.append(f"Camera {i + 1} ({display}):\n")
            parts.append(frames[cam_id])
        parts.append(self._build_overview_instruction(schema))

        from src.utils.profiler import profiler
        with profiler.measure("VLM.house_overview"):
            raw = self._client.call_parts(
                parts,
                max_tokens=self._config.overview_max_tokens,
                image_max_size=image_max_size,
                json_schema=schema,
            )

        if not raw:
            return None

        # Grammar enforcement guarantees valid JSON; parse can still fail if
        # generation was truncated by overview_max_tokens mid-document.
        try:
            data = json.loads(raw)
            summary = str(data.get("summary", "")).strip()
            camera_scenes = {k: str(v).strip() for k, v in data.get("scenes", {}).items() if v}
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"VLM overview parse error (truncated? raise overview_max_tokens): {e} | raw={raw!r}")
            return None

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

    @staticmethod
    def _build_overview_context(person_states: list[dict]) -> str:
        """Person-state header placed BEFORE the camera images.

        Explicitly framed as sensor data for the summary only, so the model
        does not substitute it for the per-camera image descriptions.
        """
        if not person_states:
            return "Sensor data: no people detected.\n"
        parts = []
        for p in person_states:
            name = p.get("name", "Unknown")
            zone = p.get("zone") or "unknown location"
            activity = p.get("activity") or ""
            gender = p.get("gender") or ""
            age = p.get("age_group") or ""
            detail = ", ".join(x for x in [activity, gender, age] if x and x != "unknown")
            parts.append(f"{name} in {zone}" + (f" ({detail})" if detail else ""))
        return (
            "Sensor data (context for the summary only): "
            + ". ".join(parts) + ".\n"
        )

    @staticmethod
    def _build_overview_instruction(schema: dict) -> str:
        """Trailing JSON instruction rendered FROM the schema.

        Field rules come from the schema's description fields (the grammar
        converter drops them, so this is how they reach the model) and the
        key order mirrors the schema property order: the model must describe
        each camera image before it may summarize.
        """
        props = schema["properties"]
        scene_props = props["scenes"]["properties"]
        scene_desc = next(iter(scene_props.values()))["description"]
        summary_desc = props["summary"]["description"]
        cam_keys = ", ".join(f'"{c}":"<one sentence>"' for c in scene_props)
        return (
            "Fields:\n"
            f'- each key of "scenes": {scene_desc}\n'
            f'- "summary": {summary_desc}\n'
            "Reply with JSON only:\n"
            f'{{"scenes":{{{cam_keys}}},'
            '"summary":"<one sentence>"}'
        )
