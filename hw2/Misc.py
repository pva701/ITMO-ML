__author__ = 'pva701'

import numpy as np
import numpy.random as rnd


def gen_random_combination(n, k, exclude_id):
    if 2 * k > n:
        perm = np.append(np.arange(0, exclude_id), np.arange(exclude_id + 1, n))
        rnd.shuffle(perm)
        return perm[:k]
    else:
        used = [False] * n
        ret = []
        while len(ret) < k:
            parent_id = rnd.randint(0, n)
            if used[parent_id] or parent_id == exclude_id:
                continue
            used[parent_id] = True
            ret.append(parent_id)
        ret = np.array(ret)
        return ret
