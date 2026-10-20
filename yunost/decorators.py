"""Небольшие декораторы, которыми пользуется остальной код."""
import functools
import logging
import time

log = logging.getLogger("yunost")


def timed(func):
    """Пишет в лог, сколько секунд работала функция."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            log.info("%s: %.2f c", func.__qualname__, time.perf_counter() - start)
    return wrapper
