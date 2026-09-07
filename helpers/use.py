"""Select which clip is live, for the animator and training."""
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")
CLIPS = os.path.join(ROOT, "animations")
sys.path.insert(0, HERE)

import ballistic
import checkclip
import dm_common as C
import writecsv


def report(path):
    """One-line physics summary: flight windows, gravity, angular momentum."""
    try:
        raw, jq, lp, rq = checkclip.load(path)
    except Exception:
        return "unreadable"
    F = len(jq)
    _, lq = C.forward_kinematics(rq, lp[:, C.ROOT], jq)
    wins = ballistic.find_flight(lp, lq)
    if not wins:
        return "grounded (no flight phase)"

    import angular
    com = C.com_of(lp)
    worst_g, worst_L, worst_ax = 0.0, 0.0, 0.0
    for (lo, hi) in wins:
        n = hi - lo + 1
        if n < 6:
            continue
        t = np.arange(n) / C.FPS
        g = 2 * np.polyfit(t, com[lo:hi + 1, 1], 2)[0]
        worst_g = max(worst_g, abs(g + 9.81))
        a = angular.audit(jq, lp[:, C.ROOT], rq, [(lo + 2, hi - 1)])[0]
        L = a["L"]
        Lm = L.mean(axis=0)
        nm = np.linalg.norm(Lm)
        if nm > 1e-6:
            cos = (L @ Lm) / (np.linalg.norm(L, axis=1) * nm + 1e-12)
            worst_ax = max(worst_ax, np.degrees(np.arccos(np.clip(cos, -1, 1))).max())
        worst_L = max(worst_L, a["pct"])
    return (f"flight x{len(wins)}  g err {worst_g:4.1f}  "
            f"|L| {worst_L:3.0f}%  axis {worst_ax:3.0f} deg")


def listing():
    names = sorted(f[:-4] for f in os.listdir(CLIPS) if f.endswith(".csv"))
    print(f"{'clip':<22}{'frames':>7}  physics")
    print("-" * 74)
    for n in names:
        p = os.path.join(CLIPS, n + ".csv")
        try:
            F = len(np.loadtxt(p, delimiter=","))
        except Exception:
            F = 0
        print(f"{n:<22}{F:>7}  {report(p)}")
    print("\ngrounded clips have no flight phase, so no angular momentum to violate")
    print("-- those are the safest to train.")


def main(name):
    path = name if os.path.exists(name) else os.path.join(CLIPS, f"{name}.csv")
    if not os.path.exists(path):
        print(f"no clip '{name}'\n")
        listing()
        sys.exit(1)

    rows = np.loadtxt(path, delimiter=",")
    shutil.copy(path, os.path.join(SRC, "baked_motion.csv"))
    writecsv.write_npz(os.path.join(SRC, "motion.npz"), rows)
    print(f"active clip: {os.path.basename(path)}  ({rows.shape[0]} frames, "
          f"{rows.shape[0] / C.FPS:.2f}s)")
    print(f"  physics: {report(path)}")
    print(f"  -> {os.path.join(SRC, 'baked_motion.csv')}   (animator)")
    print(f"  -> {os.path.join(SRC, 'motion.npz')}        (training)")
    print("\nNOTE: train.py's DPHASE and test.py's DPHASE are per-clip "
          f"-- this clip needs 1/{rows.shape[0] - 1}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        listing()
