import cython

from libcpp cimport bool
from cython.operator cimport dereference as deref
from cython cimport floating, integral
from collections import defaultdict


@cython.cdivision(True)
@cython.boundscheck(False)
def update_cooccurrence(list token_ids,
                        dict cooccur,
                        int window_size = 10,
                        bool uniform_count = False):
    """
    """

    cdef:
        int glb_end, lcl_end, row, col, cur, other
        int i, j

    glb_end = len(token_ids)
    for i in range(glb_end):
        cur = token_ids[i]

        lcl_end = min(i + window_size + 1, glb_end)
        for j in range(i + 1, lcl_end):
            other = token_ids[j]

            if cur <= other:
                row, col = cur, other
            else:
                row, col = other, cur

            # prep container if not yet made
            if row not in cooccur:
                cooccur[row] = defaultdict(float)
            if col not in cooccur[row]:
                cooccur[row][col] = 0.

            if uniform_count:
                cooccur[row][col] += 1.
            else:
                cooccur[row][col] += 1. / (j - i)
