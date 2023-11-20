import itertools

def batched(iterable, n):
    """A homegrown version of itertools.batched, taken
    directly from `itertools docs <https://docs.python.org/3/library/itertools.html#itertools.batched>`_
    for earlier versions of itertools

    Args:
        iterable (Iterable): loop over this
        n (int): batch size

    Returns:
        batch (Iterable): a batch of size n from the iterable
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
