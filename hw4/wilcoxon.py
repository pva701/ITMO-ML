import numpy as np

def wilcoxon(x, y):
    assert len(x) == len(y)
    d = x - y
    d = np.compress(np.not_equal(d, 0), d, axis=-1)
    count = len(d)
    assert count >= 10    
    r = np.sort(abs(d))
    for i in range(0, len(r)):
        j = i
        while r[i] == r[j]:
            j += 1
        r[i:j] = (j*(j-1)/2 - i*(i-1)/2) / (j - i)
    W = sum(r * np.sign(d))
    z_score = W / np.sqrt(nr*(nr+1)*(2*nr+1)/6)
    return W, z_score, 