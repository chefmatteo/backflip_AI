"""Enforce angular momentum conservation during flight."""
import numpy as np
import dm_common as C
from dm_common import qmat, qmul, qconj, qnorm, axis_angle_to_quat

FPS = C.FPS


def body_inertia():
    """Diagonal inertia in each bone's own frame, matching mma.cpp's Bone ctor."""
    I = np.zeros((14, 3))
    for i in range(14):
        m, d = C.MASS[i], C.DIMS[i]
        if C.SHAPE[i] == 1:
            f = 2 * d
            I[i] = m / 12 * np.array([f[1]**2 + f[2]**2,
                                      f[0]**2 + f[2]**2,
                                      f[0]**2 + f[1]**2])
        else:
            L, r = 2 * d[0], d[1]
            side = m / 12 * (3 * r * r + L * L)
            I[i] = [0.5 * m * r * r, side, side]
    return I


IB = body_inertia()


def _omega(lq):
    """Per-link world angular velocity by finite difference of orientation."""
    F = lq.shape[0]
    w = np.zeros((F, 14, 3))
    for f in range(1, F):
        dq = qmul(lq[f], qconj(lq[f - 1]))
        dq = np.where(dq[..., :1] < 0, -dq, dq)
        v = dq[..., 1:]
        s = np.linalg.norm(v, axis=-1, keepdims=True)
        ang = 2 * np.arctan2(s, dq[..., :1])
        w[f] = np.where(s < 1e-9, 0.0, ang * v / np.maximum(s, 1e-12)) * FPS
    w[0] = w[1]
    return w


def inertia_world(lq_f, lp_f, com_f):
    """Composite inertia tensor of the whole body about the COM, at one frame."""
    R = qmat(lq_f)
    I = np.zeros((3, 3))
    for i in range(14):
        I += R[i] @ np.diag(IB[i]) @ R[i].T
        r = lp_f[i] - com_f
        I += C.MASS[i] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return I


def total_L(lq_f, lp_f, w_f, vel_f, com_f, comv_f):
    """Total angular momentum about the COM at one frame."""
    R = qmat(lq_f)
    L = np.zeros(3)
    for i in range(14):
        L += (R[i] @ np.diag(IB[i]) @ R[i].T) @ w_f[i]
        L += C.MASS[i] * np.cross(lp_f[i] - com_f, vel_f[i] - comv_f)
    return L


def audit(joint_q, root_p, root_q, windows):
    """Report |L| variation over each flight window. Read-only."""
    lp, lq = C.forward_kinematics(root_q, root_p, joint_q)
    w = _omega(lq)
    com = C.com_of(lp)
    vel = np.gradient(lp, 1 / FPS, axis=0)
    comv = np.gradient(com, 1 / FPS, axis=0)
    out = []
    for (lo, hi) in windows:
        Ls = np.array([total_L(lq[f], lp[f], w[f], vel[f], com[f], comv[f])
                       for f in range(lo, hi + 1)])
        mag = np.linalg.norm(Ls, axis=1)
        out.append(dict(lo=lo, hi=hi, mean=mag.mean(), std=mag.std(),
                        pct=100 * mag.std() / max(mag.mean(), 1e-9), L=Ls))
    return out


def fix_rotation(joint_q, root_p, root_q, windows, verbose=True):
    """Rewrite root orientation so each flight window conserves angular momentum."""
    root_q = qnorm(root_q.copy())
    for (lo, hi) in windows:
        n = hi - lo + 1
        if n < 3:
            continue
        lp, lq = C.forward_kinematics(root_q, root_p, joint_q)
        com = C.com_of(lp)
        w = _omega(lq)
        vel = np.gradient(lp, 1 / FPS, axis=0)
        comv = np.gradient(com, 1 / FPS, axis=0)

        # Total turn the animator asked for, as a rotation vector,
        dq_tot = qmul(root_q[hi], qconj(root_q[lo]))
        if dq_tot[0] < 0:
            dq_tot = -dq_tot
        v = dq_tot[1:]
        s = np.linalg.norm(v)
        axis = v / s if s > 1e-9 else np.array([0.0, 0.0, 1.0])
        angle = 2 * np.arctan2(s, dq_tot[0])
        T = (n - 1) / FPS

        I_mean = np.mean([inertia_world(lq[f], lp[f], com[f])
                          for f in range(lo, hi + 1)], axis=0)
        L = I_mean @ (axis * angle / max(T, 1e-9))

        # Integrate w(t) = I(t)^-1 (L - L_joint(t)) forward from the takeoff pose.
        q = root_q[lo].copy()
        new_q = [q.copy()]
        for f in range(lo, hi):
            _, lqf = C.forward_kinematics(q[None], root_p[f:f + 1], joint_q[f:f + 1])
            lpf, _ = C.forward_kinematics(q[None], root_p[f:f + 1], joint_q[f:f + 1])
            comf = C.com_of(lpf)[0]
            I = inertia_world(lqf[0], lpf[0], comf)
            # angular momentum already carried by the animated joint motion
            L_joint = total_L(lqf[0], lpf[0], w[f], vel[f], comf, comv[f])
            L_spin = (qmat(lqf[0])[C.ROOT] @ np.diag(IB[C.ROOT])
                      @ qmat(lqf[0])[C.ROOT].T) @ w[f, C.ROOT]
            omega = np.linalg.solve(I, L - (L_joint - L_spin))
            dq = axis_angle_to_quat(omega / FPS)
            q = qnorm(qmul(dq, q))
            new_q.append(q.copy())
        root_q[lo:hi + 1] = np.array(new_q)
        if verbose:
            print(f"  momentum-fixed flight {lo}-{hi}: net turn "
                  f"{np.degrees(angle):.0f} deg over {T:.2f}s")
    return qnorm(root_q)


def rigid_flight(joint_q, root_p, root_q, windows, blend=10, verbose=True):
    """Make every flight phase exactly conservable, by construction."""
    from dm_common import slerp
    joint_q = qnorm(joint_q.copy())
    root_q = qnorm(root_q.copy())
    for (lo, hi) in windows:
        n = hi - lo + 1
        if n < 3:
            continue
        hold = joint_q[lo].copy()
        for f in range(lo, hi + 1):
            joint_q[f] = hold
        dq = qmul(root_q[hi], qconj(root_q[lo]))
        if dq[0] < 0:
            dq = -dq
        v = dq[1:]
        s = np.linalg.norm(v)
        axis = v / s if s > 1e-9 else np.array([0.0, 0.0, 1.0])
        angle = 2 * np.arctan2(s, dq[0])
        # A rigid body spinning at constant ANGULAR VELOCITY does not
        I0 = inertia_world(
            C.forward_kinematics(root_q[lo][None], root_p[lo:lo+1],
                                 joint_q[lo:lo+1])[1][0],
            C.forward_kinematics(root_q[lo][None], root_p[lo:lo+1],
                                 joint_q[lo:lo+1])[0][0],
            C.com_of(C.forward_kinematics(root_q[lo][None], root_p[lo:lo+1],
                                          joint_q[lo:lo+1])[0])[0])
        T = (n - 1) / FPS

        def integrate(scale):
            q = root_q[lo].copy()
            out = [q.copy()]
            L = I0 @ (axis * angle / max(T, 1e-9)) * scale
            for _ in range(n - 1):
                lpf, lqf = C.forward_kinematics(q[None], root_p[lo:lo+1],
                                                joint_q[lo:lo+1])
                I = inertia_world(lqf[0], lpf[0], C.com_of(lpf)[0])
                w = np.linalg.solve(I, L)
                q = qnorm(qmul(axis_angle_to_quat(w / FPS), q))
                out.append(q.copy())
            return np.array(out)

        # bisect on |L| so the net turn matches what the animator asked for
        best, lo_s, hi_s = integrate(1.0), 0.2, 3.0
        for _ in range(24):
            mid = 0.5 * (lo_s + hi_s)
            cand = integrate(mid)
            dqc = qmul(cand[-1], qconj(cand[0]))
            if dqc[0] < 0:
                dqc = -dqc
            got = 2 * np.arctan2(np.linalg.norm(dqc[1:]), dqc[0])
            if got < angle:
                lo_s = mid
            else:
                hi_s = mid
            best = cand
        root_q[lo:hi + 1] = best
        # Feather the frozen pose into the stance frames on BOTH
        for b in range(1, blend + 1):
            a = b / (blend + 1.0)
            a = a * a * (3 - 2 * a)                      # smoothstep, C1 at both ends
            f = hi + b
            if f < len(joint_q):
                for j in range(C.NUM_JOINTS):
                    joint_q[f, j] = slerp(hold[j][None], joint_q[f, j][None],
                                          np.array([[a]]))[0]
            f = lo - b
            if f >= 0:
                for j in range(C.NUM_JOINTS):
                    joint_q[f, j] = slerp(hold[j][None], joint_q[f, j][None],
                                          np.array([[a]]))[0]
        if verbose:
            print(f"  rigid flight {lo}-{hi}: {np.degrees(angle):.0f} deg "
                  f"at constant rate over {(n-1)/FPS:.2f}s")
    return qnorm(joint_q), qnorm(root_q)


def slow_down(joint_q, root_p, root_q, factor, verbose=True):
    """Resample a clip to run `factor` times slower (more frames, same motion)."""
    from dm_common import slerp
    F = joint_q.shape[0]
    n = int(round((F - 1) * factor)) + 1
    src = np.linspace(0, F - 1, n)
    i0 = np.floor(src).astype(int).clip(0, F - 2)
    a = (src - i0)[:, None]
    jq = np.stack([slerp(joint_q[i0, j], joint_q[i0 + 1, j], a)
                   for j in range(C.NUM_JOINTS)], axis=1)
    rq = slerp(root_q[i0], root_q[i0 + 1], a)
    rp = root_p[i0] * (1 - a) + root_p[i0 + 1] * a
    if verbose:
        print(f"  slowed {factor:.2f}x: {F} -> {n} frames "
              f"({F/FPS:.2f}s -> {n/FPS:.2f}s)")
    return qnorm(jq), rp, qnorm(rq)


def limit_joint_speed(joint_q, max_speed=13.0, iters=200, verbose=True):
    """Smooth any joint that moves faster than a PD controller could follow."""
    from dm_common import slerp
    q = qnorm(joint_q.copy())
    F = q.shape[0]
    for _ in range(iters):
        aa = C.quat_to_axis_angle(q)
        sp = np.abs(np.diff(aa, axis=0) * FPS).max(axis=-1)   # (F-1, J)
        if sp.max() <= max_speed:
            break
        for j in range(C.NUM_JOINTS):
            hot = np.where(sp[:, j] > max_speed)[0]
            for f in hot:
                for k in (f, f + 1):
                    if 0 < k < F - 1:
                        mid = slerp(q[k - 1, j][None], q[k + 1, j][None],
                                    np.array([[0.5]]))[0]
                        q[k, j] = slerp(q[k, j][None], mid[None],
                                        np.array([[0.5]]))[0]
    if verbose:
        fin = np.abs(np.diff(C.quat_to_axis_angle(q), axis=0) * FPS).max()
        print(f"  joint speed limited to {fin:.2f} rad/s (cap {max_speed})")
    return qnorm(q)
