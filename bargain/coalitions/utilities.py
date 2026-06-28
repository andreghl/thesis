import numpy as np
import itertools

def get_coalitions_encode(n_players : int = 3):
    """Return the encoding of all 2^N coalitions for N players."""
    values = [0, 1]
    return list(itertools.product(values, repeat = n_players))


def encode_coalition(coalition : list, n_depots : int):
    """Returns a binary indicator vector for a list of depots in a coalition.

    Args:
    coalition: list of depot indices to include in the coalition.
    n_depots: total number of depots (and size of the output vector).

    Returns:
        A binary array of size n_depots taking a value of 1 at the indices in the coalition.
    """
    encoding = np.zeros(n_depots, dtype = np.int32)
    coalition = np.asarray(coalition, dtype = np.int32)
    encoding[coalition] = 1

    return encoding

def yield_coalitions(players : list):
    """Generate the 2^N coalitions for a list of N players."""
    n = len(players)
    for size in range(n + 1):
        for coalition in itertools.combinations(range(n), size):
            yield list(coalition)