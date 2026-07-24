"""
utils/logger.py
==================
One shared logger factory, used by every module that logs anything
(mainly text_extractor.py and opensearch_store.py, where long-running
async operations benefit from progress logging).

Why centralize this instead of each module calling
logging.basicConfig() itself?
    logging.basicConfig() only takes effect on its FIRST call per
    process -- if two different modules each called it with different
    formats/levels, whichever ran first would silently win, and
    debugging which module "owns" the log format would be confusing.
    One factory function means one consistent format everywhere, and one
    place to change it (e.g. to adjust verbosity via LOG_LEVEL).
"""

import logging
import os

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger for `name` (typically __name__ of the
    calling module). Level is controlled by the LOG_LEVEL env var
    (default INFO) so verbosity can be adjusted without touching code."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid adding duplicate handlers on reimport
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    return logger
