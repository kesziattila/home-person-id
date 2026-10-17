"""RTSP URLs carry camera credentials; logs and CLI output must never show them.

Found 2026-10-17: deploy.sh --restart tails data/app.log, whose DEBUG GStreamer
pipeline line printed rtsp://user:password@host in full.
"""

import logging
from unittest.mock import MagicMock, patch

from src.stream.rtsp_client import RTSPClient, redact_credentials


def test_redacts_user_and_password():
    assert redact_credentials("rtspsrc location=rtsp://admin:s3cr3t@10.0.0.5:554/ch1 ! x") == \
        "rtspsrc location=rtsp://***@10.0.0.5:554/ch1 ! x"


def test_leaves_urls_without_credentials_alone():
    assert redact_credentials("rtsp://10.0.0.5:554/ch1") == "rtsp://10.0.0.5:554/ch1"
    assert redact_credentials("http://host/a@b") == "http://host/a@b"


def test_connect_logs_never_contain_the_password(caplog):
    client = RTSPClient("cam", "rtsp://admin:s3cr3t@10.0.0.5:554/ch1", use_nvdec=True)
    closed = MagicMock()
    closed.isOpened.return_value = False
    with patch("src.stream.rtsp_client.cv2.VideoCapture", return_value=closed), \
            caplog.at_level(logging.DEBUG, logger="src.stream.rtsp_client"):
        client._connect()
    assert "10.0.0.5" in caplog.text  # the URL is logged ...
    assert "s3cr3t" not in caplog.text and "admin" not in caplog.text  # ... without credentials
