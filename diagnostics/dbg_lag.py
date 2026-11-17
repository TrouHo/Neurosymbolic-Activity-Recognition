"""Is there a time offset between labels and signals? Cross-correlate walking labels with movement energy."""
import numpy as np
from emob import config as C
from emob.data.dataset import load_npz

walk = C.L2_CLASSES.index("Marcher")
lags_s = np.arange(-120, 120.5, 0.5)
for i in range(1, 9):
    d = load_npz(f"user{i}")
    T = len(d["l2"])
    # movement energy per 0.1 s step: std of phone |acc| within a 1-s window
    ph = d["phone50"][: T * 5, :3].astype(float)
    mag = np.linalg.norm(ph, axis=1)
    e = np.array([mag[max(0, 5 * t - 25):5 * t + 25].std() for t in range(T)])
    w = (d["l2"] == walk).astype(float)
    ok = d["annotated"]
    res = []
    for L in lags_s:
        k = int(round(L * 10))
        if k >= 0:
            a, b, m = w[: T - k], e[k:], ok[: T - k]
        else:
            a, b, m = w[-k:], e[: T + k], ok[-k:]
        a, b = a[m], b[m]
        res.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
    res = np.array(res)
    j = np.nanargmax(res)
    z = int(np.where(lags_s == 0)[0][0])
    print(f"user{i}: corr at lag 0 = {res[z]:.3f}; best lag = {lags_s[j]:+.1f} s (corr {res[j]:.3f}); "
          f"walk steps {int(w.sum())}; energy walk/non-walk at 0 = {e[(w==1)&ok].mean():.2f}/{e[(w==0)&ok].mean():.2f}")
