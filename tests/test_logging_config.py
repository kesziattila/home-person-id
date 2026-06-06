"""Tests for logging configuration: rotation fields and console-handler suppression.

Added with the RotatingFileHandler change (remote Jetson deployment):
- LoggingConfig gained max_bytes / backup_count, parsed via LoggingConfig(**yaml_dict)
- _setup_logging suppresses the console StreamHandler when HPID_NO_CONSOLE_LOG is set
  AND a file handler exists (detached runs redirect stdout to data/stdout.log; without
  suppression every log line would be duplicated there).
"""

import logging
import logging.handlers

from src.config import LoggingConfig


class TestLoggingConfigFields:
    def test_rotation_defaults(self):
        config = LoggingConfig()
        assert config.max_bytes == 20_000_000
        assert config.backup_count == 5

    def test_rotation_parsed_from_yaml_dict(self):
        # load_config() constructs LoggingConfig(**data["logging"])
        config = LoggingConfig(**{"level": "INFO", "file": "data/app.log",
                                  "max_bytes": 1000, "backup_count": 2})
        assert config.max_bytes == 1000
        assert config.backup_count == 2


class TestSetupLoggingHandlers:
    """Exercise the handler-selection logic of PersonIDSystem._setup_logging
    without constructing the full system (replicates its branching)."""

    @staticmethod
    def _build_handlers(logging_config, env, tmp_path):
        handlers = []
        if logging_config.file:
            handlers.append(
                logging.handlers.RotatingFileHandler(
                    tmp_path / "app.log",
                    maxBytes=logging_config.max_bytes,
                    backupCount=logging_config.backup_count,
                )
            )
        if not (handlers and env.get("HPID_NO_CONSOLE_LOG")):
            handlers.insert(0, logging.StreamHandler())
        return handlers

    def test_console_suppressed_when_detached_with_file(self, tmp_path):
        handlers = self._build_handlers(
            LoggingConfig(file="data/app.log"), {"HPID_NO_CONSOLE_LOG": "1"}, tmp_path
        )
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.handlers.RotatingFileHandler)

    def test_console_kept_when_not_detached(self, tmp_path):
        handlers = self._build_handlers(LoggingConfig(file="data/app.log"), {}, tmp_path)
        assert len(handlers) == 2
        assert isinstance(handlers[0], logging.StreamHandler)

    def test_console_kept_without_file_even_if_detached(self, tmp_path):
        # Suppression is only honored when a file handler exists — never go dark
        handlers = self._build_handlers(
            LoggingConfig(file=None), {"HPID_NO_CONSOLE_LOG": "1"}, tmp_path
        )
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)

    def test_rotation_triggers_at_max_bytes(self, tmp_path):
        log_path = tmp_path / "rot.log"
        handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=500, backupCount=2)
        logger = logging.getLogger("test_rotation")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        for i in range(50):
            logger.info("x" * 40)
        handler.close()
        logger.removeHandler(handler)
        assert log_path.exists()
        assert (tmp_path / "rot.log.1").exists()
        assert log_path.stat().st_size <= 600
