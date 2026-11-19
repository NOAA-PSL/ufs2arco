import logging

from ufs2arco.log import setup_simple_log, SimpleFormatter


def test_setup_simple_log_sets_level_and_adds_handler():
    logger = logging.getLogger("ufs2arco")
    before = list(logger.handlers)
    before_level = logger.level
    try:
        setup_simple_log(level=logging.DEBUG)
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > len(before)
        handler = logger.handlers[-1]
        assert handler.level == logging.DEBUG
        assert isinstance(handler.formatter, SimpleFormatter)
    finally:
        # restore logger to pre-test state
        for h in list(logger.handlers):
            if h not in before:
                logger.removeHandler(h)
        logger.setLevel(before_level)


def test_simple_formatter_converts_relative_created_to_seconds():
    formatter = SimpleFormatter(fmt="%(relativeCreated)d %(message)s")
    record = logging.LogRecord(
        name="ufs2arco",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.relativeCreated = 5000  # milliseconds
    out = formatter.format(record)
    # formatter divides by 1000 -> seconds
    assert out.startswith("5 ")
    assert "hello" in out
