"""Logging configuration for TR-SINDy.

All modules use ``logging.getLogger(__name__)`` so that the application can
attach handlers (console for the CLI, a Qt bridge for the GUI) without
changing call sites.  A ``NullHandler`` is installed by default so that
library use without a handler never spews "No handlers could be found".

Usage in modules::

    import logging
    log = logging.getLogger(__name__)
    log.info("processed %d frames", n)

CLI setup::

    from tr_sindy_app._logging import configure_cli_logging
    configure_cli_logging(verbose=True)

GUI setup::

    from tr_sindy_app._logging import QtLogHandler
    handler = QtLogHandler(text_widget)
    logging.getLogger("tr_sindy_app").addHandler(handler)
"""

from __future__ import annotations

import logging
import sys

_PACKAGE_LOGGER = "tr_sindy_app"

# Format used by both the CLI console handler and the GUI bridge.
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def configure_cli_logging(verbose: bool = False) -> None:
    """Attach a stderr handler to the package logger.

    Call once at CLI entry.  Idempotent — repeated calls replace the
    existing console handler rather than stacking duplicates.
    """
    root = logging.getLogger(_PACKAGE_LOGGER)
    level = logging.DEBUG if verbose else logging.INFO
    root.setLevel(level)
    # Remove any prior console handler we added.
    for h in list(root.handlers):
        if getattr(h, "_tr_sindy_console", False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATEFMT))
    handler._tr_sindy_console = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    # Don't let the root logger double-log our messages.
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a NullHandler attached (library-safe)."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    return log


class QtLogHandler(logging.Handler):
    """Forward log records to a ``QPlainTextEdit`` (the in-app log widget).

    Safe to construct on any thread; emit marshals onto the Qt event loop
    via ``QMetaObject.invokeMethod`` so it works from worker threads.
    """

    def __init__(self, text_widget) -> None:
        super().__init__()
        self._widget = text_widget
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                            _LOG_DATEFMT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from PyQt6.QtCore import Q_ARG, QMetaObject, Qt
            msg = self.format(record)
            # QPlainTextEdit.appendPlainText is thread-safe via invokeMethod.
            QMetaObject.invokeMethod(
                self._widget, "appendPlainText",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, msg),
            )
        except Exception:
            # Fallback: direct append (best-effort, may warn if cross-thread).
            try:
                self._widget.appendPlainText(self.format(record))
            except Exception:
                self.handleError(record)
