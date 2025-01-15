import sys
import logging

class SimpleFormatter(logging.Formatter):
    def format(self, record):
        record.relativeCreated = record.relativeCreated // 1000
        return super().format(record)

def setup_simple_log(level=logging.INFO):

    logging.basicConfig(
        stream=sys.stdout,
        level=level,
    )
    logger = logging.getLogger()
    formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)
