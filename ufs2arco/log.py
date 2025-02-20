import sys
import logging

logger = logging.getLogger("ufs2arco")

class SimpleFormatter(logging.Formatter):
    def format(self, record):
        record.relativeCreated = record.relativeCreated // 1000
        return super().format(record)

def setup_simple_log(level=logging.INFO):

    logger.setLevel(level=level)
    formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)-7s] %(message)s")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level=level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
