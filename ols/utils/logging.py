"""Logging utilities."""

import logging.config

from ols.app.models.config import LoggingConfig


def configure_logging(logging_config: LoggingConfig) -> None:
    """Configure application logging according to the configuration."""
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "root": {  # root logger
            "level": logging_config.library_log_level,
            "handlers": ["console"],
        },
        "loggers": {
            "ols": {
                "level": logging_config.app_log_level,
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(name)s:%(filename)s:%(lineno)d] %(levelname)s: %(message)s"  # noqa E501
            },
        },
    }

    logging.config.dictConfig(logging_config)
