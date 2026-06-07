"""Tests for grammar-enforced JSON output of the VLM house overview.

Background (live bug, 2026-06-06): llama.cpp (b8233) silently skips
OpenAI-style response_format constraints when the request contains images —
the model then wraps its reply in ```json fences and json.loads fails
("VLM overview parse error" warnings on the Jetson). The fix sends llama.cpp's
top-level `json_schema` body param via extra_body, which is converted to a
GBNF grammar and enforced during sampling even for multimodal requests.
"""

import json
from unittest.mock import MagicMock

import numpy as np

from src.config import VLMConfig
from src.vlm.vlm_analyzer import VLMAnalyzer, _build_overview_schema
from src.vlm.vlm_client import VLMClient


class TestBuildOverviewSchema:
    def test_every_camera_is_a_required_scene_key(self):
        schema = _build_overview_schema(["ground_floor", "hallway"])
        scenes = schema["properties"]["scenes"]
        assert set(scenes["properties"].keys()) == {"ground_floor", "hallway"}
        assert scenes["required"] == ["ground_floor", "hallway"]
        # The model must not invent cameras or extra top-level keys
        assert scenes["additionalProperties"] is False
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["scenes", "summary"]

    def test_scenes_precede_summary_in_schema_order(self):
        """Grammar follows property order — scenes MUST come before summary.

        With summary-first, the person-state context dominated generation and
        the model filled every scene key by repeating its own summary (live
        incident 2026-06-07: all cameras said 'A person is in the work room.').
        """
        schema = _build_overview_schema(["cam1"])
        assert list(schema["properties"].keys()) == ["scenes", "summary"]

    def test_person_context_precedes_images_in_parts(self):
        """The sensor-data header must be the FIRST part, before any image —
        adjacent to the JSON instruction it leaks into the scene fields."""
        analyzer = VLMAnalyzer(VLMConfig(enabled=True))
        analyzer._client = MagicMock()
        analyzer._client.call_parts.return_value = (
            '{"scenes":{"ground_floor":"a room"},"summary":"ok"}'
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        analyzer.generate_house_overview(
            [{"name": "Ati", "zone": "work_room"}], {"ground_floor": frame}
        )
        parts = analyzer._client.call_parts.call_args.args[0]
        assert isinstance(parts[0], str) and "Sensor data" in parts[0]
        assert "Ati in work_room" in parts[0]
        # Instruction is last and asks for scenes before summary
        assert '"scenes":{' in parts[-1]
        assert parts[-1].index('"scenes"') < parts[-1].index('"summary"')

    def test_schema_is_json_serializable(self):
        json.dumps(_build_overview_schema(["cam1"]))


class TestJsonSchemaKwargs:
    def test_schema_sent_as_llamacpp_top_level_param(self):
        # MUST go through extra_body["json_schema"], NOT response_format —
        # response_format is ignored by llama.cpp for multimodal requests.
        schema = {"type": "object"}
        kwargs = VLMClient._json_schema_kwargs(schema)
        assert kwargs == {"extra_body": {"json_schema": schema}}
        assert "response_format" not in kwargs

    def test_no_schema_means_no_extra_kwargs(self):
        assert VLMClient._json_schema_kwargs(None) == {}


class TestGenerateHouseOverview:
    def _analyzer_with_mock_client(self, raw_response):
        analyzer = VLMAnalyzer(VLMConfig(enabled=True))
        analyzer._client = MagicMock()
        analyzer._client.call_parts.return_value = raw_response
        return analyzer

    def _frames(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        return {"ground_floor": frame, "hallway": frame}

    def test_valid_json_produces_overview(self):
        raw = json.dumps({
            "summary": "A person is in the living room.",
            "scenes": {"ground_floor": "Empty room.", "hallway": "A person standing."},
        })
        analyzer = self._analyzer_with_mock_client(raw)
        overview = analyzer.generate_house_overview([], self._frames())
        assert overview is not None
        assert overview.summary == "A person is in the living room."
        assert overview.camera_scenes == {
            "ground_floor": "Empty room.",
            "hallway": "A person standing.",
        }

    def test_overview_call_passes_schema_for_all_cameras(self):
        analyzer = self._analyzer_with_mock_client('{"summary": "x", "scenes": {}}')
        analyzer.generate_house_overview([], self._frames())
        kwargs = analyzer._client.call_parts.call_args.kwargs
        schema = kwargs["json_schema"]
        assert set(schema["properties"]["scenes"]["required"]) == {"ground_floor", "hallway"}

    def test_fenced_output_is_rejected_not_crashing(self):
        # Defense in depth: if the server ever regresses to fenced output,
        # we log + return None (deliberately NOT stripping fences).
        raw = '```json\n{"summary": "x", "scenes": {}}\n```'
        analyzer = self._analyzer_with_mock_client(raw)
        assert analyzer.generate_house_overview([], self._frames()) is None

    def test_truncated_json_returns_none(self):
        raw = '{"summary": "A person is in the liv'
        analyzer = self._analyzer_with_mock_client(raw)
        assert analyzer.generate_house_overview([], self._frames()) is None
