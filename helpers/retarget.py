"""Retarget a DeepMimic clip onto the mma skeleton."""
import json
import numpy as np
import dm_common as C
from dm_common import qmul, qconj, qnorm, axis_angle_to_quat, slerp

# --- DeepMimic humanoid3d.txt frame layout ------------------------------------- [0] duration, [1:4] root
DM_LAYOUT = [("chest", 4), ("neck", 4),
             ("right_hip", 4), ("right_knee", 1), ("right_ankle", 4),
             ("right_shoulder", 4), ("right_elbow", 1),
             ("left_hip", 4), ("left_knee", 1), ("left_ankle", 4),
             ("left_shoulder", 4), ("left_elbow", 1)]

# --- rest-frame reconciliation ------------------------------------------------- mma.cpp bakes each limb's hanging direction
REST_AA = np.zeros((13, 3))
REST_AA[3] = REST_AA[4] = [0.0, 0.0, -np.pi / 2]    # shoulders
REST_AA[7] = REST_AA[8] = [0.0, 0.0, -np.pi / 2]    # hips
REST_AA[11] = REST_AA[12] = [0.0, 0.0, np.pi / 2]   # ankles
REST_Q = axis_angle_to_quat(REST_AA)                # (13,4)

# Root translation is scaled by the leg-length ratio so foot
LEG_MMA = C.HIP_Y - C.ANKLE_Y
LEG_DM = 0.4215 + 0.4099
ROOT_SCALE = LEG_MMA / LEG_DM

DOWN = axis_angle_to_quat(np.array([0.0, 0.0, -np.pi / 2]))
DOWN_INV = qconj(DOWN)

def load_dm(path):
    """Read a humanoid3d clip -> (root_p, root_q, joints dict, per-frame duration)."""
    with open(path) as f:
        d = json.load(f)
    frames = np.array(d["Frames"], dtype=np.float64)
    dur = frames[:, 0]
    root_p = frames[:, 1:4].copy()
    root_q = qnorm(frames[:, 4:8].copy())
    joints, col = {}, 8
    for name, width in DM_LAYOUT:
        joints[name] = frames[:, col:col + width].copy()
        col += width
    return root_p, root_q, joints, dur, d.get("Loop", "none")


def _limb(q_dm, j):
    """DM spherical limb rotation -> mma joint quat, including mma's rest offset."""
    local = qmul(qmul(DOWN_INV, qnorm(q_dm)), DOWN)
    return qnorm(qmul(REST_Q[j], local))


def _hinge(angle, axis, j):
    """DM revolute angle -> mma joint quat about `axis`, on top of the rest offset."""
    q = axis_angle_to_quat(angle.reshape(-1, 1) * np.array(axis)[None, :])
    return qnorm(qmul(REST_Q[j], q))


def retarget(path):
    """-> (joint_q (F,13,4), root_p (F,3), root_q (F,4), loop flag)."""
    root_p, root_q, J, dur, loop = load_dm(path)
    F = root_p.shape[0]
    jq = np.zeros((F, 13, 4))
    jq[:, :, 0] = 1.0

    # --- torso: split DM's single chest rotation over mma's waist + spine ---
    chest = qnorm(J["chest"])
    ident = np.tile([1.0, 0, 0, 0], (F, 1))
    half = slerp(ident, chest, 0.5)
    jq[:, 0] = half                                  # waist  (pelvis -> abs)
    jq[:, 1] = qnorm(qmul(qconj(half), chest))       # spine  (abs -> chest), so the
    #                                                  composition still equals `chest`
    jq[:, 2] = qnorm(J["neck"])                      # neck is torso-frame in both

    # --- arms. DM shoulder is relative to chest, same parent as mma's. ---
    jq[:, 3] = _limb(J["right_shoulder"], 3)
    jq[:, 4] = _limb(J["left_shoulder"], 4)
    # elbow: DM revolute about its local x; in mma's rotated limb frame that is +z.
    # DM elbow angle is negative-flexion, mma's forearm folds the same way about +z.
    jq[:, 5] = _hinge(J["right_elbow"][:, 0], [0, 0, 1], 5)
    jq[:, 6] = _hinge(J["left_elbow"][:, 0], [0, 0, 1], 6)

    # --- legs ---
    jq[:, 7] = _limb(J["right_hip"], 7)
    jq[:, 8] = _limb(J["left_hip"], 8)
    jq[:, 9] = _hinge(J["right_knee"][:, 0], [0, 0, 1], 9)
    jq[:, 10] = _hinge(J["left_knee"][:, 0], [0, 0, 1], 10)
    jq[:, 11] = _limb(J["right_ankle"], 11)
    jq[:, 12] = _limb(J["left_ankle"], 12)

    rp = root_p * ROOT_SCALE
    return qnorm(jq), rp, qnorm(root_q), dur, loop


def resample(jq, rp, rq, dur, fps=C.FPS):
    """DeepMimic clips are authored at their own per-frame duration; mma bakes a
    strict 30 Hz grid, so resample onto it (slerp rotations, lerp translation)."""
    t_src = np.concatenate([[0.0], np.cumsum(dur[:-1])])
    total = t_src[-1]
    n = int(np.floor(total * fps)) + 1
    t_dst = np.arange(n) / fps
    idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(t_src) - 2)
    span = np.maximum(t_src[idx + 1] - t_src[idx], 1e-9)
    a = ((t_dst - t_src[idx]) / span).clip(0, 1)[:, None]

    rp_o = rp[idx] * (1 - a) + rp[idx + 1] * a
    rq_o = slerp(rq[idx], rq[idx + 1], a)
    jq_o = np.stack([slerp(jq[idx, j], jq[idx + 1, j], a) for j in range(13)], axis=1)
    return qnorm(jq_o), rp_o, qnorm(rq_o)


def mirror_pitch(joint_q, root_p, root_q):
    """Turn a backward rotation into a forward one (backflip -> front somersault)."""
    jq = joint_q.copy()
    rq = root_q.copy()
    rp = root_p.copy()
    rp[:, 0] *= -1.0                      # forward axis flips
    rq[:, 2] *= -1.0                      # y component
    rq[:, 3] *= -1.0                      # z component (pitch) -> rotation reverses
    jq[:, :, 2] *= -1.0
    jq[:, :, 3] *= -1.0
    return qnorm(jq), rp, qnorm(rq)
