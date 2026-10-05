"""Video tier: one handheld walkthrough clip -> the LiDAR tier's plan pipeline.

No depth, no poses, no IMU in the file, so both are estimated:

1. frames sampled at <= MAX_FPS (and <= MAX_FRAMES overall)
2. metric depth per frame from Depth Anything V2 (metric indoor)
3. visual odometry: ORB matches to the previous frame, lifted to 3D with that
   frame's depth, PnP-RANSAC for the new pose
4. scale consistency: each frame's depth is rescaled to agree with the map at
   its PnP inliers; at the end the map is rescaled so the median frame keeps
   its native model scale (no single frame anchors the scale)
5. gravity: "up" is the dominant floor/ceiling normal, seeded by image-up
   (the protocol holds the phone upright)
6. drift correction, room segmentation, walls, openings: shared with LiDAR
7. intervals widened by the video error model (VIDEO_REL, VIDEO_ABS)
"""
from pathlib import Path

import numpy as np

from roomscan import drift as drift_mod
from roomscan.depth_model import predict
from roomscan.lidar_io import CV_TO_ARKIT, Frame, camera_points
from roomscan.output import assemble, render, widen
from roomscan.plan2d import Cloud, build_plan

MAX_FPS = 6.0
MAX_FRAMES = 600
WORK_W = 640                 # long side used for features
FX_RATIO = 0.82              # iPhone 1x video: fx ~ 0.80-0.84 x long side
DEPTH_SIZE = (256, 192)      # same grid as the LiDAR depth maps
# error model, 90% half-widths added in quadrature to the geometric fit:
# monocular depth scale (~3%) and focal-length prior; calibrate on the benchmark
VIDEO_REL = 0.03
VIDEO_ABS = 0.02


def read_frames(video: Path, max_fps=MAX_FPS, max_frames=MAX_FRAMES):
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 6
    every = max(1, int(round(fps / max_fps)), -(-n // max_frames))
    out, i = [], 0
    while True:
        if not cap.grab():
            break
        if i % every == 0:
            ok, img = cap.retrieve()
            if ok:
                h, w = img.shape[:2]
                s = WORK_W / max(h, w)
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                out.append((i, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    return out, fps


def _K(w, h, fx_ratio):
    f = fx_ratio * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def _sample(depth, uv, w, h):
    dh, dw = depth.shape
    c = np.clip((uv[:, 0] * dw / w).astype(int), 0, dw - 1)
    r = np.clip((uv[:, 1] * dh / h).astype(int), 0, dh - 1)
    return depth[r, c]


def visual_odometry(images, depths, K):
    """Camera-to-world poses (OpenCV axes) for every frame, and which frames tracked.

    Each frame is matched to the previous one: ORB matches lifted to 3D with the
    previous frame's depth, PnP-RANSAC for the new pose. Depths keep their own
    (calibrated) scale -- chaining scale corrections compounds bias. A frame
    that fails to track holds the previous pose, so one blurred frame or fast
    turn costs that frame, not the rest of the walk.
    """
    import cv2

    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    h, w = images[0].shape[:2]
    feats = [orb.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None) for im in images]
    poses, tracked = [np.eye(4)], [True]
    for j in range(1, len(images)):
        i = j - 1
        T = _pnp(feats[i], feats[j], depths[i], poses[i], K, w, h, bf)
        tracked.append(T is not None)
        poses.append(T if T is not None else poses[i].copy())
    return poses, np.array(tracked)


def _pnp(fi, fj, depth_i, pose_i, K, w, h, bf):
    import cv2

    (kp_i, d_i), (kp_j, d_j) = fi, fj
    if d_i is None or d_j is None:
        return None
    m = bf.match(d_i, d_j)
    if len(m) < 40:
        return None
    uv_i = np.float32([kp_i[x.queryIdx].pt for x in m])
    uv_j = np.float32([kp_j[x.trainIdx].pt for x in m])
    z = _sample(depth_i, uv_i, w, h)
    good = (z > 0.2) & (z < 8.0)
    if good.sum() < 30:
        return None
    uv_i, uv_j, z = uv_i[good], uv_j[good], z[good]
    Xc = np.stack([(uv_i[:, 0] - K[0, 2]) / K[0, 0] * z, (uv_i[:, 1] - K[1, 2]) / K[1, 1] * z, z], 1)
    Xw = Xc @ pose_i[:3, :3].T + pose_i[:3, 3]
    ok, rvec, tvec, inl = cv2.solvePnPRansac(Xw.astype(np.float64), uv_j.astype(np.float64), K, None,
                                             reprojectionError=3.0, iterationsCount=300, confidence=0.999)
    if not ok or inl is None or len(inl) < 25:
        return None
    R, _ = cv2.Rodrigues(rvec)
    Tcw = np.eye(4)
    Tcw[:3, :3], Tcw[:3, 3] = R, tvec[:, 0]
    Twc = np.linalg.inv(Tcw)
    if np.linalg.norm(Twc[:3, 3] - pose_i[:3, 3]) > 0.6:      # implausible between samples
        return None
    return Twc


def _up_vector(cam):
    """Dominant floor/ceiling normal (world), seeded by the first camera's image-up."""
    n = np.concatenate([c[1] for c in cam])
    u = np.array([0.0, 1.0, 0.0])
    for cone in (0.5, 0.8, 0.95):
        d = n @ u
        sel = np.abs(d) > cone
        if sel.sum() < 100:
            break
        v = n[sel] * np.sign(d[sel])[:, None]
        u = v.mean(0)
        u /= np.linalg.norm(u)
    return u


def _rot_to(u, target=np.array([0.0, 1.0, 0.0])):
    v = np.cross(u, target)
    c = float(u @ target)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def process(capture: Path, out_dir: Path, drift: bool = True, fx_ratio: float = FX_RATIO,
            damage: bool = True, **_) -> dict:
    capture = Path(capture)
    samples, fps = read_frames(capture)
    if len(samples) < 10:
        raise RuntimeError("video too short")
    images = [im for _, im in samples]
    h, w = images[0].shape[:2]
    K = _K(w, h, fx_ratio)
    depths = [predict(im, DEPTH_SIZE) for im in images]
    poses_cv, tracked = visual_odometry(images, depths, K)
    Kd = K.copy()
    Kd[:2] *= DEPTH_SIZE[0] / w
    # re-express in a world whose axes are the first camera's ARKit axes, so the
    # first frame's image-up is world +y (the seed for gravity below)
    frames = [Frame(samples[j][0], CV_TO_ARKIT @ poses_cv[j] @ CV_TO_ARKIT, Kd, depths[j], None, None)
              for j in range(len(images)) if tracked[j]]
    cam = [camera_points(f, step=2) for f in frames]
    # gravity alignment
    world_n = [(None, n @ f.pose[:3, :3].T) for f, (_, n) in zip(frames, cam)]
    up = _up_vector(world_n)
    A = np.eye(4)
    A[:3, :3] = _rot_to(up)
    for f in frames:
        f.pose = A @ f.pose
    poses, drift_info = drift_mod.correct([f.pose for f in frames], cam, enabled=drift)

    pts, nrm, src = [], [], []
    for i, ((p, n), T) in enumerate(zip(cam, poses)):
        pts.append(p @ T[:3, :3].T + T[:3, 3])
        nrm.append(n @ T[:3, :3].T)
        src.append(np.full(len(p), i, np.int32))
    pts, nrm, src = np.concatenate(pts), np.concatenate(nrm), np.concatenate(src)
    cloud = Cloud(np.stack([pts[:, 0], -pts[:, 2]], 1).astype(np.float64), pts[:, 1].astype(np.float64),
                  np.stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]], 1).astype(np.float64), src)
    cams = np.array([[T[0, 3], -T[2, 3]] for T in poses])
    rooms, adjacency, theta, _, floor = build_plan(cloud, cams, np.array([f.index for f in frames]))
    if not rooms:
        raise RuntimeError("no rooms recovered from video")

    drift_info["vo_frames_tracked"] = f"{int(tracked.sum())}/{len(samples)}"
    doc = assemble(capture, "video", "iPhone (video, no depth)", rooms, adjacency, drift_info)
    if damage:
        from roomscan.damage import View, run
        rgb_of = {samples[j][0]: samples[j][1] for j in range(len(samples))}
        views = [View(rgb_of[f.index], f.depth, f.K, T) for f, T in zip(frames, poses)]
        run(doc, views, rooms, theta, floor)
    widen(doc, VIDEO_REL, VIDEO_ABS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render(doc, out_dir / "plan.png")
    return doc
