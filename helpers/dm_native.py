"""Parse DeepMimic humanoid3d clip files."""
import json
import numpy as np

FPS = 30.0

with open("clips/humanoid3d.txt") as f:
    _CH = json.load(f)

JOINTS = _CH["Skeleton"]["Joints"]
BODIES = _CH["BodyDefs"]
NJ = len(JOINTS)
PARENT = [j["Parent"] for j in JOINTS]
ATTACH = np.array([[j["AttachX"], j["AttachY"], j["AttachZ"]] for j in JOINTS])
MASS = np.array([b["Mass"] for b in BODIES])
# body attach offset relative to its joint, plus shape params
BOFF = np.array([[b["AttachX"], b["AttachY"], b["AttachZ"]] for b in BODIES])
SHAPE = [b["Shape"] for b in BODIES]
PARAM = np.array([[b["Param0"], b["Param1"], b["Param2"]] for b in BODIES])

LAYOUT = [("chest", 4), ("neck", 4), ("right_hip", 4), ("right_knee", 1),
          ("right_ankle", 4), ("right_shoulder", 4), ("right_elbow", 1),
          ("left_hip", 4), ("left_knee", 1), ("left_ankle", 4),
          ("left_shoulder", 4), ("left_elbow", 1)]
NAME2ID = {j["Name"]: j["ID"] for j in JOINTS}


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1)


def qconj(q): return q * np.array([1.0, -1, -1, -1])
def qnorm(q): return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def qmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
        np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
        np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


def aa2q(aa):
    t = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(t < 1e-9, 1.0, t)
    return np.where(t < 1e-9, np.array([1.0, 0, 0, 0]),
                    np.concatenate([np.cos(t/2), aa/safe*np.sin(t/2)], -1))


def load(path):
    with open(path) as f:
        d = json.load(f)
    fr = np.array(d["Frames"], dtype=np.float64)
    dur = fr[:, 0]
    root_p, root_q = fr[:, 1:4].copy(), qnorm(fr[:, 4:8].copy())
    jq = np.tile(np.array([1.0, 0, 0, 0]), (len(fr), NJ, 1))
    col = 8
    for name, w in LAYOUT:
        i = NAME2ID[name]
        if w == 4:
            jq[:, i] = qnorm(fr[:, col:col+4])
        else:
            jq[:, i] = aa2q(fr[:, col:col+1] * np.array([0.0, 0, 1.0]))
        col += w
    return root_p, root_q, jq, dur, d.get("Loop", "none")


def fk(root_p, root_q, jq):
    """World position/orientation of every BODY (not joint)."""
    F = root_p.shape[0]
    wq = np.zeros((F, NJ, 4)); wp = np.zeros((F, NJ, 3))
    wq[:, 0] = root_q; wp[:, 0] = root_p
    for i in range(1, NJ):
        p = PARENT[i]
        wq[:, i] = qmul(wq[:, p], jq[:, i])
        R = qmat(wq[:, p])
        wp[:, i] = wp[:, p] + np.einsum('fab,b->fa', R, ATTACH[i])
    # body centres sit at BOFF in their joint's frame
    bp = np.zeros((F, NJ, 3))
    for i in range(NJ):
        R = qmat(wq[:, i])
        bp[:, i] = wp[:, i] + np.einsum('fab,b->fa', R, BOFF[i])
    return bp, wq, wp


def inertia_body():
    I = np.zeros((NJ, 3))
    for i in range(NJ):
        m, p = MASS[i], PARAM[i]
        if SHAPE[i] == "sphere":
            r = p[0] / 2
            I[i] = 2.0/5.0*m*r*r
        elif SHAPE[i] == "box":
            f = p
            I[i] = m/12*np.array([f[1]**2+f[2]**2, f[0]**2+f[2]**2, f[0]**2+f[1]**2])
        else:  # capsule: Param0 = radius, Param1 = height
            r, h = p[0]/2, p[1]
            side = m/12*(3*r*r+h*h)
            I[i] = [side, 0.5*m*r*r, side]
    return I


IB = inertia_body()


def lowest_y(bp, wq):
    """Lowest world point over the two ankle bodies."""
    out = np.full(bp.shape[0], np.inf)
    for name in ("right_ankle", "left_ankle"):
        i = NAME2ID[name]
        R = qmat(wq[:, i]); p = PARAM[i]
        if SHAPE[i] == "box":
            drop = (np.abs(R[..., 0, 1])*p[0]/2 + np.abs(R[..., 1, 1])*p[1]/2
                    + np.abs(R[..., 2, 1])*p[2]/2)
        else:
            drop = np.abs(R[..., 1, 1])*p[1]/2 + p[0]/2
        out = np.minimum(out, bp[:, i, 1] - drop)
    return out


def audit(path, air_eps=0.02, min_flight=4):
    root_p, root_q, jq, dur, loop = load(path)
    F = len(root_p)
    bp, wq, _ = fk(root_p, root_q, jq)
    com = (bp * MASS[None, :, None]).sum(1) / MASS.sum()
    low = lowest_y(bp, wq)

    # per-frame dt varies; resample uniformly for finite differences
    t = np.concatenate([[0.0], np.cumsum(dur[:-1])])
    dt = np.gradient(t)
    w = np.zeros((F, NJ, 3))
    for f in range(1, F):
        dq = qmul(wq[f], qconj(wq[f-1]))
        dq = np.where(dq[..., :1] < 0, -dq, dq)
        v = dq[..., 1:]; s = np.linalg.norm(v, axis=-1, keepdims=True)
        w[f] = np.where(s < 1e-9, 0.0,
                        2*np.arctan2(s, dq[..., :1])*v/np.maximum(s, 1e-12)) / max(dt[f], 1e-9)
    w[0] = w[1]
    vel = np.gradient(bp, axis=0) / dt[:, None, None]
    comv = np.gradient(com, axis=0) / dt[:, None]

    clear = low > air_eps
    wins, f = [], 0
    while f < F:
        if clear[f]:
            s0 = f
            while f < F and clear[f]:
                f += 1
            if f - s0 >= min_flight:
                wins.append((s0, f-1))
        else:
            f += 1

    res = dict(frames=F, dur=t[-1], loop=loop, flights=[], low_min=low.min())
    for (lo, hi) in wins:
        if hi - lo + 1 < 6:
            continue
        Ls = []
        for k in range(lo+2, hi):
            R = qmat(wq[k]); L = np.zeros(3)
            for i in range(NJ):
                L += (R[i] @ np.diag(IB[i]) @ R[i].T) @ w[k, i]
                L += MASS[i]*np.cross(bp[k, i]-com[k], vel[k, i]-comv[k])
            Ls.append(L)
        if len(Ls) < 3:
            continue
        Ls = np.array(Ls); mag = np.linalg.norm(Ls, axis=1)
        Lm = Ls.mean(0); nm = np.linalg.norm(Lm)
        drift = np.degrees(np.arccos(np.clip((Ls@Lm)/(mag*nm+1e-12), -1, 1))).max() if nm > 1e-6 else 0.0
        tt = t[lo:hi+1] - t[lo]
        gfit = 2*np.polyfit(tt, com[lo:hi+1, 1], 2)[0]
        res['flights'].append(dict(lo=lo, hi=hi, g=gfit, Lmag=mag.mean(),
                                   Lpct=100*mag.std()/max(mag.mean(), 1e-9),
                                   drift=drift))
    return res
