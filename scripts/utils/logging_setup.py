import logging
import json
import os
import sys
from datetime import datetime, timezone


class _JsonlHandler(logging.FileHandler):
    """Writes one JSON line per log record to logs/{log_file}.jsonl."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = json.dumps({
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "extra": {
                    k: v for k, v in record.__dict__.items()
                    if k not in logging.LogRecord.__dict__ and k not in (
                        "msg", "args", "levelname", "levelno", "pathname",
                        "filename", "module", "exc_info", "exc_text",
                        "stack_info", "lineno", "funcName", "created",
                        "msecs", "relativeCreated", "thread", "threadName",
                        "processName", "process", "name", "message",
                    )
                },
            })
            self.stream.write(line + "\n")
            self.stream.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str, log_file: str) -> logging.Logger:
    """Return a logger with a console handler and a JSON-lines file handler."""
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)

    jsonl_path = os.path.join("logs", f"{log_file}.jsonl")
    file_handler = _JsonlHandler(jsonl_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger
