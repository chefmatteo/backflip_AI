"""
Blender-side exporter: dump QuickRigCharacter animation from an FBX.

Run with:
  blender --background --python helpers/digital_stunts_export_fbx.py -- \
      <in.fbx> <out.npz>
"""
import sys
from pathlib import Path

import bpy
import numpy as np


BONES = [
    "QuickRigCharacter_Hips",
    "QuickRigCharacter_Spine",
    "QuickRigCharacter_Spine1",
    "QuickRigCharacter_Spine2",
    "QuickRigCharacter_Neck",
    "QuickRigCharacter_Head",
    "QuickRigCharacter_RightShoulder",
    "QuickRigCharacter_RightArm",
    "QuickRigCharacter_RightForeArm",
    "QuickRigCharacter_RightHand",
    "QuickRigCharacter_LeftShoulder",
    "QuickRigCharacter_LeftArm",
    "QuickRigCharacter_LeftForeArm",
    "QuickRigCharacter_LeftHand",
    "QuickRigCharacter_RightUpLeg",
    "QuickRigCharacter_RightLeg",
    "QuickRigCharacter_RightFoot",
    "QuickRigCharacter_RightToeBase",
    "QuickRigCharacter_LeftUpLeg",
    "QuickRigCharacter_LeftLeg",
    "QuickRigCharacter_LeftFoot",
    "QuickRigCharacter_LeftToeBase",
]


def _argv_after_double_dash():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.actions) + list(bpy.data.armatures) + list(bpy.data.meshes):
        bpy.data.batch_remove([block])


def find_armature():
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("No armature found after FBX import")
    # Prefer the one that has QuickRig bones
    for a in arms:
        names = {b.name for b in a.pose.bones}
        if "QuickRigCharacter_Hips" in names:
            return a
    return arms[0]


def quat_wxyz(q):
    # Blender Quaternion is (w, x, y, z)
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float64)


def main():
    args = _argv_after_double_dash()
    if len(args) < 2:
        raise SystemExit("usage: blender --background --python digital_stunts_export_fbx.py -- in.fbx out.npz")
    fbx_path = Path(args[0]).resolve()
    out_path = Path(args[1]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    bpy.ops.import_scene.fbx(
        filepath=str(fbx_path),
        automatic_bone_orientation=False,
        ignore_leaf_bones=False,
        force_connect_children=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
    )

    arm = find_armature()
    bpy.context.view_layer.objects.active = arm
    missing = [n for n in BONES if n not in arm.pose.bones]
    if missing:
        present = sorted(b.name for b in arm.pose.bones)
        raise RuntimeError(f"Missing bones {missing}; have {present}")

    scene = bpy.context.scene
    # Use action frame range if present
    act = arm.animation_data.action if arm.animation_data else None
    if act is not None:
        start = int(round(act.frame_range[0]))
        end = int(round(act.frame_range[1]))
    else:
        start, end = int(scene.frame_start), int(scene.frame_end)

    fps = float(scene.render.fps) / float(scene.render.fps_base)
    frames = list(range(start, end + 1))
    F = len(frames)
    N = len(BONES)

    local_q = np.zeros((F, N, 4), dtype=np.float64)
    world_p = np.zeros((F, N, 3), dtype=np.float64)
    world_q = np.zeros((F, N, 4), dtype=np.float64)
    local_q[..., 0] = 1.0
    world_q[..., 0] = 1.0

    for i, f in enumerate(frames):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        for j, name in enumerate(BONES):
            pb = arm.pose.bones[name]
            # Animation delta in bone local space (rest-relative)
            local_q[i, j] = quat_wxyz(pb.matrix_basis.to_quaternion().normalized())
            mw = arm.matrix_world @ pb.matrix
            loc = mw.to_translation()
            rot = mw.to_quaternion().normalized()
            world_p[i, j] = (loc.x, loc.y, loc.z)
            world_q[i, j] = quat_wxyz(rot)

    np.savez_compressed(
        out_path,
        bone_names=np.array(BONES),
        frames=np.array(frames, dtype=np.int32),
        fps=np.array(fps, dtype=np.float64),
        local_q=local_q,
        world_p=world_p,
        world_q=world_q,
        source=str(fbx_path),
    )
    print(f"wrote {out_path}  F={F} fps={fps} bones={N} frames={start}..{end}")


if __name__ == "__main__":
    main()
