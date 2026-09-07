"""The mma skeleton plus quaternion and FK helpers."""
import numpy as np

# ---------------- mma.cpp skeleton (Skeleton::build) ---------------- bones = { footR, footL,
NUM_LINKS, NUM_JOINTS = 14, 13
FPS, DT = 30.0, 1.0 / 30.0

BONE_NAMES = ["footR", "footL", "calfR", "calfL", "thighR", "thighL", "pelvis",
              "abs", "chest", "armR", "armL", "forearmR", "forearmL", "head"]
MASS = np.array([1., 1, 2.5, 2.5, 4.5, 4.5, 5, 7, 8, 1.8, 1.8, 1.2, 1.2, 3])
# half-extents / (half-length, radius) exactly as passed to Bone(...)
DIMS = np.array([
    [0.1187, 0.0264, 0.0440], [0.1187, 0.0264, 0.0440],   # feet   (box)
    [0.1187, 0.0572, 0.0], [0.1187, 0.0572, 0.0],         # calves (capsule)
    [0.1231, 0.0594, 0.0], [0.1231, 0.0594, 0.0],         # thighs
    [0.0, 0.0967, 0.0], [0.0, 0.0769, 0.0], [0.0, 0.1165, 0.0],   # pelvis/abs/chest
    [0.0835, 0.0440, 0.0], [0.0835, 0.0440, 0.0],         # upper arms
    [0.0659, 0.0396, 0.0], [0.0659, 0.0396, 0.0],         # forearms
    [0.0, 0.0989, 0.0],                                   # head
])
SHAPE = np.array([1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])  # 1 box, 2 capsule

# joints[] order in mma.cpp, as (child_bone, parent_bone) indices into bones[]
JOINTS = [(7, 6), (8, 7), (13, 8), (9, 8), (10, 8), (11, 9), (12, 10),
          (4, 6), (5, 6), (2, 4), (3, 5), (0, 2), (1, 3)]
JOINT_NAMES = ["waist", "spine", "neck", "shoulderR", "shoulderL", "elbowR", "elbowL",
               "hipR", "hipL", "kneeR", "kneeL", "ankleR", "ankleL"]
ROOT = 6

GAP = 0.0132
ANKLE_Y, KNEE_Y, HIP_Y, HIP_Z = 0.0528, 0.4484, 0.8880, 0.0879
SHOULDER_Y, SHOULDER_Z = 1.3716, 0.1671
PELVIS_Y = 0.9584


def _top(i):  return np.array([-DIMS[i][0] - DIMS[i][1] - GAP, 0.0, 0.0])
def _bot(i):  return np.array([ DIMS[i][0] + DIMS[i][1] + GAP, 0.0, 0.0])

# anchorA (on parent) / anchorB (on child), in the SAME order as JOINTS above.
ANCHOR_A = np.array([
    [0.0,  0.0791, 0.0],                    # pelvis -> abs
    [0.0,  0.0791, 0.0],                    # abs    -> chest
    [0.0,  0.1275, 0.0],                    # chest  -> head
    [0.0,  0.0703,  SHOULDER_Z],            # chest  -> armR
    [0.0,  0.0703, -SHOULDER_Z],            # chest  -> armL
    _bot(9), _bot(10),                      # arm    -> forearm
    [0.0, -0.0703,  HIP_Z - 0.0220],        # pelvis -> thighR
    [0.0, -0.0703, -HIP_Z + 0.0220],        # pelvis -> thighL
    _bot(4), _bot(5),                       # thigh  -> calf
    _bot(2), _bot(3),                       # calf   -> foot
])
ANCHOR_B = np.array([
    [0.0, -0.0791, 0.0],
    [0.0, -0.1055, 0.0],
    [0.0, -0.0923, 0.0],
    _top(9), _top(10),
    _top(11), _top(12),
    _top(4), _top(5),
    _top(2), _top(3),
    [-0.0308, 0.0110, 0.0], [-0.0308, 0.0110, 0.0],
])

# Rest orientation of each bone in mma.cpp: limbs get `down` = angleAxis(-pi/2, z),
# the torso chain stays identity. This is the frame joint quats are expressed in.
_DOWN_IDX = {2, 3, 4, 5, 9, 10, 11, 12}   # calves, thighs, arms, forearms


# ---------------- quaternion helpers (w,x,y,z), matching train.py ----------------
def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1, -1, -1])


def qnorm(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def qrot(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return qmul(qmul(q, qv), qconj(q))[..., 1:]


def axis_angle_to_quat(aa):
    """(...,3) axis*angle -> (...,4) quaternion, matching glm::angleAxis."""
    t = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(t < 1e-8, 1.0, t)
    axis = aa / safe
    half = t / 2.0
    return np.where(t < 1e-8,
                    np.array([1.0, 0, 0, 0]),
                    np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1))


def quat_to_axis_angle(q):
    """Inverse of the above; matches mma.cpp's load() reconstruction exactly."""
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(s, q[..., :1])
    return np.where(s < 1e-8, np.zeros_like(v), ang * v / np.maximum(s, 1e-12))


def qmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],     -1),
        np.stack([2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],     -1),
        np.stack([2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)], -1),
    ], -2)


def slerp(q0, q1, t):
    q0, q1 = qnorm(q0), qnorm(q1)
    d = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d).clip(0, 1.0)
    ang = np.arccos(d)
    s = np.sin(ang)
    lin = q0 * (1 - t) + q1 * t                      # nearly parallel -> lerp
    sl = (q0 * np.sin((1 - t) * ang) + q1 * np.sin(t * ang)) / np.maximum(s, 1e-9)
    return qnorm(np.where(s < 1e-6, lin, sl))


def forward_kinematics(root_q, root_p, joint_q):
    """FK exactly as mma.cpp's Skeleton::updateKinematics does it."""
    F = joint_q.shape[0]
    link_q = np.zeros((F, NUM_LINKS, 4))
    link_p = np.zeros((F, NUM_LINKS, 3))
    link_q[:, ROOT] = root_q
    link_p[:, ROOT] = root_p
    for j, (child, parent) in enumerate(JOINTS):
        link_q[:, child] = qmul(link_q[:, parent], joint_q[:, j])
        anchor_world_a = link_p[:, parent] + qrot(link_q[:, parent], ANCHOR_A[j])
        link_p[:, child] = anchor_world_a - qrot(link_q[:, child], ANCHOR_B[j])
    return link_p, link_q


def lowest_y(link_p, link_q):
    """Per-frame lowest point over the two feet, matching Bone::lowestY()."""
    out = np.full(link_p.shape[0], np.inf)
    for i in (0, 1):
        R = qmat(link_q[:, i])
        d = DIMS[i]
        if SHAPE[i] == 1:
            drop = (np.abs(R[..., 0, 1]) * d[0] + np.abs(R[..., 1, 1]) * d[1]
                    + np.abs(R[..., 2, 1]) * d[2])
        else:
            drop = np.abs(R[..., 0, 1]) * d[0] + d[1]
        out = np.minimum(out, link_p[:, i, 1] - drop)
    return out


def com_of(link_p):
    return (link_p * MASS[None, :, None]).sum(1) / MASS.sum()
