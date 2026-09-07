"""Make a clip's flight phases physically reachable."""
import numpy as np
import dm_common as C

G = -9.81
AIR_EPS = 0.02      # m above the floor before a foot counts as clear
MIN_FLIGHT = 3      # frames; shorter than this is contact jitter, not a jump
# a window is ballistic flight only if the body really leaves the floor
MIN_LAUNCH_VY = 0.8     # m/s of measured upward COM velocity at takeoff
MIN_CLEARANCE = 0.06    # m of peak foot clearance


def find_flight(link_p, link_q):
    """Frames where BOTH feet are clear of the floor, as [lo,hi] inclusive windows."""
    clear = C.lowest_y(link_p, link_q) > AIR_EPS
    out, f, F = [], 0, len(clear)
    while f < F:
        if clear[f]:
            s = f
            while f < F and clear[f]:
                f += 1
            if f - s >= MIN_FLIGHT:
                out.append((s, f - 1))
        else:
            f += 1
    return out


def launch_velocity(com, lo, hi):
    """Recover the launch velocity the airborne COM path implies."""
    n = hi - lo + 1
    t = np.arange(n) / C.FPS
    if n < 3:
        return 0.0, 0.0, 0.0
    A = np.stack([np.ones(n), t], axis=1)
    vy = float(np.linalg.lstsq(A, com[lo:hi + 1, 1] - 0.5 * G * t * t, rcond=None)[0][1])
    vx = float(np.linalg.lstsq(A, com[lo:hi + 1, 0], rcond=None)[0][1])
    vz = float(np.linalg.lstsq(A, com[lo:hi + 1, 2], rcond=None)[0][1])
    return vx, vy, vz


def _blend(n):
    """Smoothstep 0->1 over n samples, C1 at both ends."""
    if n <= 0:
        return np.zeros(0)
    t = (np.arange(n) + 1) / (n + 1)
    return t * t * (3 - 2 * t)


def fix_translation(root_p, link_p, link_q, blend=5, verbose=True,
                    preserve_landing=False, max_takeoff=3.5):
    """Rewrite root translation so every flight window is a true ballistic arc."""
    root_p = root_p.copy()
    com0 = C.com_of(link_p)
    # root and COM differ by a per-frame offset set purely by the pose; the pose is
    # not being changed, so shifting the root by d shifts the COM by exactly d.
    offset = com0 - root_p
    com = com0.copy()
    reports = []

    for (lo, hi) in find_flight(link_p, link_q):
        n = hi - lo + 1
        t = np.arange(n) / C.FPS
        T = t[-1] if n > 1 else 1.0 / C.FPS

        vx, vy, vz = launch_velocity(com, lo, hi)
        if preserve_landing:
            vy = (com[hi, 1] - com[lo, 1] - 0.5 * G * T * T) / T
        vy = float(np.clip(vy, -max_takeoff, max_takeoff))
        y0 = com[lo, 1]

        new = np.stack([com[lo, 0] + vx * t,
                        y0 + vy * t + 0.5 * G * t * t,
                        com[lo, 2] + vz * t], axis=1)
        delta = new - com[lo:hi + 1]

        # apply, then feather the correction into the neighbouring stance frames so
        # velocity stays continuous across takeoff and landing
        com[lo:hi + 1] += delta
        pre = min(blend, lo)
        if pre:
            com[lo - pre:lo] += delta[0][None, :] * _blend(pre)[:, None]
        post = min(blend, len(com) - 1 - hi)
        if post:
            com[hi + 1:hi + 1 + post] += delta[-1][None, :] * _blend(post)[::-1, None]

        apex = new[:, 1].max() - y0
        reports.append(dict(lo=lo, hi=hi, frames=n, dur=T, vy=vy, apex=apex,
                            moved=float(np.abs(delta).max())))
        if verbose:
            print(f"  flight {lo:3d}-{hi:3d} ({n:2d}f, {T:.2f}s): "
                  f"vy0 {vy:5.2f} m/s, apex {apex:.3f} m, "
                  f"root moved up to {np.abs(delta).max():.3f} m")

    if not reports and verbose:
        print("  no flight phase (fully grounded clip) - nothing to correct")
    return root_p + (com - com0), reports


def ground_clamp(root_p, joint_q, root_q):
    """Lift any frame whose foot is under the floor, as mma.cpp's clampAboveGround"""
    link_p, link_q = C.forward_kinematics(root_q, root_p, joint_q)
    low = C.lowest_y(link_p, link_q)
    return root_p + np.stack([np.zeros_like(low), np.maximum(-low, 0.0),
                              np.zeros_like(low)], axis=1)


def retime_flight(joint_q, root_p, root_q, link_p, link_q, max_takeoff=3.5,
                  verbose=True):
    """Compress or stretch each flight window to the hang time physics actually"""
    from dm_common import slerp
    com = C.com_of(link_p)
    windows = find_flight(link_p, link_q)
    if not windows:
        return joint_q, root_p, root_q

    keep = np.ones(len(joint_q), dtype=bool)
    pieces = []   # (insert_after_index, joint_q, root_p, root_q) for rebuilt windows
    for (lo, hi) in windows:
        _, vy, _ = launch_velocity(com, lo, hi)
        vy = float(np.clip(vy, -max_takeoff, max_takeoff))
        # time to return to the takeoff COM height: t = 2*vy/g
        T_real = max(2.0 * vy / (-G), 1.0 / C.FPS)
        n_real = max(int(round(T_real * C.FPS)), 2)
        n_auth = hi - lo + 1
        # a clip whose hang time already matches its launch needs no retiming; only
        # rewrite timing when the authored air time is clearly unaffordable
        if abs(n_real - n_auth) <= max(2, int(0.15 * n_auth)):
            continue
        if verbose:
            print(f"  retime flight {lo}-{hi}: {n_auth} frames "
                  f"({n_auth / C.FPS:.2f}s) -> {n_real} frames ({n_real / C.FPS:.2f}s)"
                  f"  [vy0 {vy:.2f} m/s]")
        # resample the authored airborne poses onto the shorter/longer window
        src = np.linspace(0, n_auth - 1, n_real)
        i0 = np.floor(src).astype(int).clip(0, n_auth - 2)
        a = (src - i0)[:, None]
        w_jq = np.stack([slerp(joint_q[lo + i0, j], joint_q[lo + i0 + 1, j], a)
                         for j in range(C.NUM_JOINTS)], axis=1)
        w_rq = slerp(root_q[lo + i0], root_q[lo + i0 + 1], a)
        w_rp = root_p[lo + i0] * (1 - a) + root_p[lo + i0 + 1] * a
        keep[lo:hi + 1] = False
        pieces.append((lo, w_jq, w_rp, w_rq))

    # rebuild the clip: untouched frames in order, each window's resample spliced in
    out_jq, out_rp, out_rq = [], [], []
    done = set()
    for f in range(len(joint_q)):
        for (lo, w_jq, w_rp, w_rq) in pieces:
            if f == lo and lo not in done:
                out_jq.append(w_jq); out_rp.append(w_rp); out_rq.append(w_rq)
                done.add(lo)
        if keep[f]:
            out_jq.append(joint_q[f][None]); out_rp.append(root_p[f][None])
            out_rq.append(root_q[f][None])
    return (np.concatenate(out_jq), np.concatenate(out_rp), np.concatenate(out_rq))
