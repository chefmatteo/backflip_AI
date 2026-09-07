"""Report whether a clip is physically learnable."""
import sys
import numpy as np
import dm_common as C
from ballistic import find_flight
import angular

G = -9.81
TOL_G = 2.5        # m/s^2 tolerance on implied gravity
TOL_HORIZ = 1.5    # m/s^2 tolerance on in-flight horizontal accel
MAX_JOINT_SPEED = 15.0   # rad/s (~860 deg/s); above this no PD controller can track


def load(path):
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim == 1:
        raw = raw[None, :]
    F = raw.shape[0]
    jq = raw[:, 0:52].reshape(F, 13, 4)
    lp = raw[:, 52:94].reshape(F, 14, 3)
    rq = raw[:, 137:141] if raw.shape[1] >= 141 else np.tile([1., 0, 0, 0], (F, 1))
    return raw, jq, lp, rq


def check(path, verbose=True):
    raw, jq, lp, rq = load(path)
    F = lp.shape[0]
    _, lq = C.forward_kinematics(rq, lp[:, C.ROOT], jq)
    com = C.com_of(lp)
    ok = True
    print(f"\n=== {path}  ({F} frames, {F / C.FPS:.2f}s) ===")

    low = C.lowest_y(lp, lq)
    if low.min() < -0.02:
        print(f"  [WARN] foot penetrates floor by {-low.min():.3f} m "
              f"at frame {int(low.argmin())}")

    flights = find_flight(lp, lq)
    if not flights:
        print("  grounded clip, no flight phase to check")
    for (lo, hi) in flights:
        n = hi - lo + 1
        t = np.arange(n) / C.FPS
        y = com[lo:hi + 1, 1]
        ph_lo, ph_hi = lo / max(F - 1, 1), hi / max(F - 1, 1)
        print(f"  flight frames {lo}-{hi} (phase {ph_lo:.3f}-{ph_hi:.3f}, {n} frames)")
        if n < 3:
            continue
        g_fit = 2 * np.polyfit(t, y, 2)[0]
        resid = np.abs(y - np.polyval(np.polyfit(t, y, 2), t)).max()
        vx = np.gradient(com[lo:hi + 1, 0], 1 / C.FPS)
        vz = np.gradient(com[lo:hi + 1, 2], 1 / C.FPS)
        ax = np.abs(np.gradient(vx, 1 / C.FPS)).max()
        az = np.abs(np.gradient(vz, 1 / C.FPS)).max()
        bad_g = abs(g_fit - G) > TOL_G
        bad_h = max(ax, az) > TOL_HORIZ
        ok &= not (bad_g or bad_h)
        print(f"    implied g   {g_fit:7.2f} m/s^2   {'FAIL' if bad_g else 'ok'}"
              f"   (want {G})")
        print(f"    parabola residual {resid:.4f} m")
        print(f"    horiz accel |ax| {ax:5.2f} |az| {az:5.2f}   "
              f"{'FAIL' if bad_h else 'ok'}   (want ~0)")

    # angular momentum: airborne there is no external torque, so total
    for (lo, hi) in flights:
        if hi - lo + 1 < 6:
            continue
        a = angular.audit(jq, lp[:, C.ROOT], rq, [(lo + 2, hi - 1)])[0]
        L = a['L']
        # Conservation is a statement about the VECTOR, not just its
        Lm = L.mean(axis=0)
        nm = np.linalg.norm(Lm)
        if nm > 1e-6:
            cosang = (L @ Lm) / (np.linalg.norm(L, axis=1) * nm + 1e-12)
            drift = np.degrees(np.arccos(np.clip(cosang, -1, 1))).max()
        else:
            drift = 0.0
        bad_L = a['pct'] > 15.0 or drift > 15.0
        ok &= not bad_L
        print(f"    angular momentum |L| {a['mean']:6.2f} +/- {a['pct']:4.1f}%   "
              f"axis drift {drift:4.1f} deg   {'FAIL' if bad_L else 'ok'}")

    # Joint speed: a reference the actuators cannot follow is unreachable
    aa = C.quat_to_axis_angle(jq)
    if F > 1:
        jvv = np.abs(np.diff(aa, axis=0) * C.FPS)
        jv = jvv.max()
        j_worst = int(jvv.max(axis=(0, 2)).argmax())
        f_worst = int(jvv[:, j_worst, :].max(axis=1).argmax()) + 1
        bad_v = jv > MAX_JOINT_SPEED
        ok &= not bad_v
        print(f"  peak joint speed {jv:6.2f} rad/s ({np.degrees(jv):4.0f} deg/s) "
              f"at {C.JOINT_NAMES[j_worst]} f{f_worst}   "
              f"{'FAIL' if bad_v else 'ok'}   (limit {MAX_JOINT_SPEED})")

    print(f"  --> {'PASS: physically trackable' if ok else 'FAIL: not trackable, fix before training'}")
    return ok


if __name__ == "__main__":
    paths = sys.argv[1:] or ["../src/baked_motion.csv"]
    results = [check(p) for p in paths]
    sys.exit(0 if all(results) else 1)
