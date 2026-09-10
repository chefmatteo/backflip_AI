"""Convert Digital Stunts QuickRig export (.npz) into mma animations/*.csv."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dm_common as C
from dm_common import qmul, qconj, qnorm, qrot
import writecsv
import ballistic

# indices into the exporter bone list
IDX = {
    "Hips": 0,
    "Spine": 1,
    "Spine1": 2,
    "Spine2": 3,
    "Neck": 4,
    "Head": 5,
    "RightShoulder": 6,
    "RightArm": 7,
    "RightForeArm": 8,
    "RightHand": 9,
    "LeftShoulder": 10,
    "LeftArm": 11,
    "LeftForeArm": 12,
    "LeftHand": 13,
    "RightUpLeg": 14,
    "RightLeg": 15,
    "RightFoot": 16,
    "RightToeBase": 17,
    "LeftUpLeg": 18,
    "LeftLeg": 19,
    "LeftFoot": 20,
    "LeftToeBase": 21,
}

# Same rest offsets retarget.py uses for spherical limbs — unused for Blender
# matrix_basis (already rest-relative). Kept as documentation of the DM path.


def _get(local_q, name):
    return local_q[:, IDX[name]]


def _compose(a, b):
    return qnorm(qmul(a, b))


def map_joints(local_q):
    """QuickRig local deltas -> mma joint_q (F,13,4)."""
    from retarget import REST_Q, DOWN, DOWN_INV

    F = local_q.shape[0]
    jq = np.zeros((F, 13, 4))
    jq[:, :, 0] = 1.0

    # torso: fold spine chain into waist + spine
    jq[:, 0] = _get(local_q, "Spine")
    jq[:, 1] = _compose(_get(local_q, "Spine1"), _get(local_q, "Spine2"))
    jq[:, 2] = _get(local_q, "Neck")

    # Blender/Mixamo bones are Y-along; mma spherical limbs are X-along after REST.
    # Same DOWN conjugation DeepMimic retarget uses.
    def _limb(q, j):
        local = qnorm(qmul(qmul(DOWN_INV, qnorm(q)), DOWN))
        return qnorm(qmul(REST_Q[j], local))

    jq[:, 3] = _limb(_compose(_get(local_q, "RightShoulder"), _get(local_q, "RightArm")), 3)
    jq[:, 4] = _limb(_compose(_get(local_q, "LeftShoulder"), _get(local_q, "LeftArm")), 4)
    jq[:, 5] = qnorm(qmul(REST_Q[5], _get(local_q, "RightForeArm")))
    jq[:, 6] = qnorm(qmul(REST_Q[6], _get(local_q, "LeftForeArm")))
    jq[:, 7] = _limb(_get(local_q, "RightUpLeg"), 7)
    jq[:, 8] = _limb(_get(local_q, "LeftUpLeg"), 8)
    jq[:, 9] = qnorm(qmul(REST_Q[9], _get(local_q, "RightLeg")))
    jq[:, 10] = qnorm(qmul(REST_Q[10], _get(local_q, "LeftLeg")))
    jq[:, 11] = _limb(_get(local_q, "RightFoot"), 11)
    jq[:, 12] = _limb(_get(local_q, "LeftFoot"), 12)
    return qnorm(jq)


def rebuild_root_from_bones(world_p):
    """Root quat from hips/head/thighs — more stable than Mixamo hip world_q."""
    hips = world_p[:, IDX["Hips"]]
    head = world_p[:, IDX["Head"]]
    rup = world_p[:, IDX["RightUpLeg"]]
    lup = world_p[:, IDX["LeftUpLeg"]]

    body_up = head - hips
    body_up /= np.maximum(np.linalg.norm(body_up, axis=-1, keepdims=True), 1e-9)

    # +Z = right in mma (see hip anchors)
    right = rup - lup
    right = right - (right * body_up).sum(-1, keepdims=True) * body_up
    right /= np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-9)

    forward = np.cross(body_up, right)
    forward /= np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-9)
    right = np.cross(forward, body_up)  # re-orthogonalize

    F = len(world_p)
    rq = np.zeros((F, 4))
    for i in range(F):
        R = np.stack([forward[i], body_up[i], right[i]], axis=0)
        rq[i] = _rotmat_to_quat(R)
    return _hemi_fix(qnorm(rq))


def strip_yaw(rq):
    """Keep lean (tilt) but lock facing to +X — for in-place carve demos."""
    # Decompose: yaw about Y, then tilt. facing = qrot(rq, +X) projected to XZ.
    forward = qrot(rq, np.broadcast_to(np.array([1.0, 0.0, 0.0]), (len(rq), 3)))
    yaw = np.arctan2(forward[:, 2], forward[:, 0])  # 0 = +X
    # undo yaw: rotate by -yaw about Y
    half = (-yaw / 2.0)[:, None]
    q_unyaw = np.concatenate([np.cos(half), np.zeros_like(half), np.sin(half), np.zeros_like(half)], axis=1)
    return _hemi_fix(qnorm(qmul(q_unyaw, rq)))


def resample_to_fps(jq, rp, rq, src_fps, dst_fps=C.FPS):
    if abs(src_fps - dst_fps) < 1e-6:
        return jq, rp, rq
    F = jq.shape[0]
    t_src = np.arange(F) / src_fps
    total = t_src[-1]
    n = int(np.floor(total * dst_fps)) + 1
    t_dst = np.arange(n) / dst_fps
    idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, F - 2)
    span = np.maximum(t_src[idx + 1] - t_src[idx], 1e-9)
    a = ((t_dst - t_src[idx]) / span).clip(0, 1)[:, None]
    from dm_common import slerp
    rp_o = rp[idx] * (1 - a) + rp[idx + 1] * a
    rq_o = slerp(rq[idx], rq[idx + 1], a)
    jq_o = np.stack([slerp(jq[idx, j], jq[idx + 1, j], a) for j in range(13)], axis=1)
    return qnorm(jq_o), rp_o, qnorm(rq_o)


def _hemi_fix(q):
    """Flip quaternion signs so consecutive frames stay in the same hemisphere."""
    out = q.copy()
    if out.ndim == 2:  # (F, 4)
        for t in range(1, len(out)):
            if np.dot(out[t], out[t - 1]) < 0:
                out[t] = -out[t]
    else:  # (F, J, 4)
        for t in range(1, len(out)):
            dots = np.sum(out[t] * out[t - 1], axis=-1)
            out[t] = np.where(dots[:, None] < 0, -out[t], out[t])
    return out


def _joint_ang_vel(jq, fps):
    """Angular velocity from quat derivative (avoids axis-angle wrap spikes)."""
    F = jq.shape[0]
    av = np.zeros((F, 13, 3))
    q0 = jq[:-1]
    q1 = jq[1:].copy()
    d = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    dq = qmul(qconj(q0), q1)
    av[1:] = 2.0 * dq[..., 1:] * fps
    return av


def write_clip(path, joint_q, root_p, root_q):
    """Like writecsv.write, but joint_av from quat deltas (stable near ±π)."""
    F = joint_q.shape[0]
    joint_q, root_q = qnorm(joint_q), qnorm(root_q)
    link_p, _ = C.forward_kinematics(root_q, root_p, joint_q)
    com = C.com_of(link_p)
    av = _joint_ang_vel(joint_q, C.FPS)

    rlv = np.zeros((F, 3))
    rlv[1:] = (root_p[1:] - root_p[:-1]) * C.FPS
    rav = np.zeros((F, 3))
    dq = qmul(root_q[1:], qconj(root_q[:-1]))
    dq = np.where(dq[:, :1] < 0, -dq, dq)
    from dm_common import quat_to_axis_angle
    rav[1:] = quat_to_axis_angle(dq) * C.FPS

    prot = writecsv.unwrap_pelvis_rot(root_q)
    flag = np.ones((F, 1))
    out = np.concatenate([
        joint_q.reshape(F, 52), link_p.reshape(F, 42), av.reshape(F, 39),
        com, flag, root_q, prot, rlv, rav,
    ], axis=1)
    assert out.shape[1] == 150, out.shape
    np.savetxt(path, out, delimiter=",", fmt="%.6g")
    return out


def smooth_jq(jq, passes=2, alpha=0.35):
    """Light temporal slerp smoothing to knock down spike frames."""
    from dm_common import slerp
    out = jq.copy()
    for _ in range(passes):
        nxt = out.copy()
        for t in range(1, len(out) - 1):
            mid = slerp(out[t - 1], out[t + 1], 0.5)
            nxt[t] = slerp(out[t], mid, alpha)
        out = qnorm(nxt)
    return out


def _rotmat_to_quat(R):
    """3x3 rotation matrix -> (w,x,y,z) quaternion."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return qnorm(np.array([w, x, y, z], dtype=np.float64))


def align_world_to_mma(world_p, world_q):
    """Map Digital Stunts / Blender world into mma Y-up, facing +X.

    The flat-5 FBX keeps the skier's head along roughly +Z and the slope along -Y,
    so treating Blender Y as up lays the figure on its side. Rebuild a basis from
    body-up (hips->head) and horizontal travel, then rotate positions + root quats.
    """
    hips = world_p[:, IDX["Hips"]]
    head = world_p[:, IDX["Head"]]
    body_up = (head - hips).mean(axis=0)
    nu = np.linalg.norm(body_up)
    if nu < 1e-8:
        raise RuntimeError("degenerate body-up from hips/head")
    body_up /= nu

    travel = hips[-1] - hips[0]
    forward = travel - body_up * np.dot(travel, body_up)
    nf = np.linalg.norm(forward)
    if nf < 1e-6:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, body_up)) > 0.9:
            tmp = np.array([0.0, 0.0, 1.0])
        forward = tmp - body_up * np.dot(tmp, body_up)
        nf = np.linalg.norm(forward)
    forward /= nf

    # mma: +Y up, +X forward. Right-handed: X×Y=Z.
    right = np.cross(forward, body_up)
    right /= np.linalg.norm(right)
    forward = np.cross(body_up, right)
    forward /= np.linalg.norm(forward)
    R = np.stack([forward, body_up, right], axis=0)
    assert np.linalg.det(R) > 0.9, f"align basis not right-handed det={np.linalg.det(R)}"
    qR = _rotmat_to_quat(R)

    world_p = np.einsum("ij,...j->...i", R, world_p)
    F = world_q.shape[0]
    qR_b = np.broadcast_to(qR, (F, 4))
    world_q = qnorm(qmul(qR_b[:, None, :], world_q))
    print(f"align: body_up_src={body_up}  forward_src={forward}  "
          f"up·Y after={(R @ body_up)[1]:.3f}  fwd·X after={(R @ (hips[-1]-hips[0]))[0]:.3f}")
    return world_p, world_q


def convert(npz_path, csv_path, max_seconds=None, ground=True, lo=None, hi=None,
            in_place=False):
    data = np.load(npz_path, allow_pickle=True)
    local_q = data["local_q"]
    world_p = data["world_p"]
    world_q = data["world_q"]
    fps = float(data["fps"])

    world_p, world_q = align_world_to_mma(world_p, world_q)

    # Scale bone world positions first so root rebuild + leg length agree
    mid_foot = 0.5 * (world_p[:, IDX["RightFoot"]] + world_p[:, IDX["LeftFoot"]])
    leg = np.linalg.norm(world_p[:, IDX["Hips"]] - mid_foot, axis=-1)
    leg_med = float(np.median(leg))
    leg_mma = C.HIP_Y - C.ANKLE_Y
    if leg_med > 1e-6:
        scale = leg_mma / leg_med
        world_p = world_p * scale
        print(f"scale={scale:.4f} (src leg={leg_med:.3f} -> {leg_mma:.3f})")

    jq = map_joints(local_q)
    jq = _hemi_fix(jq)
    jq = smooth_jq(jq)
    rp = world_p[:, IDX["Hips"]].copy()
    rq = rebuild_root_from_bones(world_p)
    rq = smooth_jq(rq[:, None, :], passes=2, alpha=0.25)[:, 0]

    # Center XZ at origin on first frame; keep Y for ground snap
    rp[:, [0, 2]] -= rp[0, [0, 2]]

    jq, rp, rq = resample_to_fps(jq, rp, rq, fps, C.FPS)
    jq = _hemi_fix(jq)
    rq = _hemi_fix(rq)

    if lo is not None or hi is not None:
        a = int(lo or 0)
        b = int(hi if hi is not None else len(jq))
        jq, rp, rq = jq[a:b], rp[a:b], rq[a:b]
        print(f"cut frames [{a}:{b}] -> {len(jq)} frames")
        rp[:, [0, 2]] -= rp[0, [0, 2]]

    if max_seconds is not None:
        n = int(max_seconds * C.FPS)
        jq, rp, rq = jq[:n], rp[:n], rq[:n]

    if in_place:
        # Stay put: no glide. Freeze facing to upright +X — strip_yaw fails when the
        # Mixamo root is tumbled (forward isn't horizontal), which looked like spinning.
        rp[:, 0] = 0.0
        rp[:, 2] = 0.0
        rq = np.zeros_like(rq)
        rq[:, 0] = 1.0  # identity: upright, facing +X
        print("in-place: root XZ pinned + root orient frozen upright (joints carry carve)")

    if ground:
        # Place the lowest point on the floor every frame (lift OR drop).
        link_p, link_q = C.forward_kinematics(rq, rp, jq)
        low = C.lowest_y(link_p, link_q)
        rp = rp.copy()
        rp[:, 1] -= low

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_clip(csv_path, jq, rp, rq)
    peak = np.linalg.norm(_joint_ang_vel(jq, C.FPS).reshape(-1, 3), axis=-1).max()
    print(f"wrote {csv_path}  frames={jq.shape[0]}  duration={jq.shape[0]/C.FPS:.2f}s  peak|av|={peak:.2f} rad/s")
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("csv")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--lo", type=int, default=None, help="first frame (after resample)")
    ap.add_argument("--hi", type=int, default=None, help="end frame exclusive (after resample)")
    ap.add_argument("--in-place", action="store_true",
                    help="pin root XZ so the figure carves in place")
    ap.add_argument("--no-ground", action="store_true")
    args = ap.parse_args()
    convert(args.npz, args.csv, max_seconds=args.max_seconds, ground=not args.no_ground,
            lo=args.lo, hi=args.hi, in_place=args.in_place)


if __name__ == "__main__":
    main()
