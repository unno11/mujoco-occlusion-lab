"""Dev helper: probe candidate camera poses and check the occlusion geometry.

Renders one PNG per candidate into results/camprobe/, and prints, for each
candidate, the z-band that a plate at the given y would need in order to hide
the box (bottom / top of the box as seen at the plate plane).

    MUJOCO_GL=egl ./.venv/bin/python scripts/dev_camera_probe.py
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "panda" / "scene_push.xml"
OUT = ROOT / "results" / "camprobe"

CANDS = [
    # (pos, target, fovy, plate_y)
    ((0.40, -1.15, 0.82), (0.50, 0.05, 0.15), 60, -0.12),
    ((0.35, -0.95, 0.95), (0.50, 0.05, 0.10), 65, -0.12),
    ((0.50, -1.30, 0.70), (0.48, 0.05, 0.20), 55, -0.12),
    ((0.20, -0.90, 0.85), (0.52, 0.05, 0.12), 62, -0.12),
]


def look_at(pos, target, up=(0.0, 0.0, 1.0)):
    pos = np.asarray(pos, float)
    d = np.asarray(target, float) - pos
    d /= np.linalg.norm(d)
    z = -d
    x = np.cross(d, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1)  # columns
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def band_at_plane(cam_pos, cam_quat, plate_y, pts):
    """z of the rays (cam -> pts) where they cross y=plate_y, per point."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, cam_quat)
    out = []
    for p in pts:
        p = np.asarray(p, float)
        d = p - cam_pos
        t = (plate_y - cam_pos[1]) / d[1]
        out.append((t, cam_pos + t * d))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    cid = model.camera("rgb").id
    box = data.geom("box").xpos
    box_top = box + np.array([0, 0, 0.028])
    box_bot = box - np.array([0, 0, 0.028])
    tip = 0.5 * (data.body("left_finger").xpos + data.body("right_finger").xpos)

    renderer = mujoco.Renderer(model, height=480, width=640)
    for i, (pos, target, fovy, plate_y) in enumerate(CANDS):
        model.cam_pos[cid] = pos
        model.cam_quat[cid] = look_at(pos, target)
        model.cam_fovy[cid] = fovy
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cid)
        frame = renderer.render()
        import imageio.v2 as imageio

        p = OUT / f"cam{i}.png"
        imageio.imwrite(p, frame)

        print(f"--- cam{i} pos={pos} fovy={fovy}")
        for name, pt in (("box_bottom", box_bot), ("box_top", box_top), ("box_mid", box), ("ee_tip", tip)):
            t, hit = band_at_plane(np.array(pos), np.array(model.cam_quat[cid]), plate_y, [pt])[0]
            print(f"    {name:11s} at y={plate_y}: z={hit[2]:.3f} x={hit[0]:.3f} (t={t:.2f})")
    renderer.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
