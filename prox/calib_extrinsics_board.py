"""10-2 카메라 위치 보정 (자동판): 바닥에 놓은 체커보드 하나로 두 카메라의 위치·방향을 한 번에 계산
calib_extrinsics.py(테이프 기준점 줄자 측정 + 웹 클릭)를 대신한다. 재는 일도 클릭도 없다.
사용법: 체커보드를 두 카메라에 모두 보이는 바닥에 평평하게 놓고
        python calib_extrinsics_board.py --cams "$CAM_URL" http://<1호기>:8140/stream.mjpg
좌표계: 원점 = 보드 첫 내부 코너 바로 아래 바닥(--board-height), x = 보드 가로(코너 9개 방향), y = 보드 세로, z = 위(카메라 쪽 +), 단위 m
결과: calib/cam0.npz, calib/cam1.npz (calib_extrinsics.py 와 같은 형식), calib/world_points.json (방 범위),
      results/10_extrinsics.jsonl (카메라 위치, 재투영 오차, 두 카메라 삼각측량 검증 오차)

고려한 점
- 두 카메라가 코너 순서를 서로 반대로(180도 돌려) 읽으면 좌표계가 어긋난다. 첫 칸(코너 0·1·cols·cols+1 사이)이
  어두운 쪽을 기준으로 순서를 맞춘다 (9x6 보드는 180도 돌리면 첫 칸 색이 바뀐다).
- 보드가 멀면 칸이 작아 검출이 어렵다. 2배 확대 후 findChessboardCornersSB 로 찾는다.
- 정지 상태라 여러 프레임에서 찾은 코너를 평균해 잡음을 줄인다.
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lab"))
from camera import LatestFrame, mask   # noqa: E402
from stereo import load_cam, undistort, triangulate   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cams", nargs=2, required=True, help="cam0, cam1 영상 주소")
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square", type=float, default=0.025, help="칸 한 변(m). 크게 출력했으면 실제 잰 값")
ap.add_argument("--board-height", type=float, default=0.0,
                help="보드를 바닥이 아닌 책상 등에 놓았을 때 그 높이(m). 바닥이 z=0 이 되도록 좌표를 올린다 (ankle1 모드는 발목 z=0 을 가정)")
ap.add_argument("--frames", type=int, default=10, help="카메라마다 평균낼 검출 횟수")
ap.add_argument("--timeout", type=float, default=60, help="카메라마다 검출을 기다릴 최대 시간(초)")
ap.add_argument("--room", type=float, nargs=4, default=[-1.5, -1.5, 2.5, 2.5], metavar=("X0", "Y0", "X1", "Y1"),
                help="proximity_app 이 쓰는 방 범위(m, 보드 좌표계)")
ap.add_argument("--res", default="640x480")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
PAT = (args.cols, args.rows)
CALIB = Path("calib"); CALIB.mkdir(exist_ok=True)
RESULTS = Path(__file__).resolve().parent.parent / "results"; RESULTS.mkdir(exist_ok=True)


def detect(gray):
    """2배 확대해서 찾고 원래 좌표로 되돌린다 (멀리 있는 작은 보드 대비)"""
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    ok, c = cv2.findChessboardCornersSB(big, PAT, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        return None
    return c.reshape(-1, 2) / 2.0


def canonical(gray, c):
    """첫 칸이 어두운 쪽이 코너 0 이 되도록 순서를 맞춘다 → 두 카메라가 같은 좌표계를 쓴다"""
    def quad_mean(i):
        q = c[[i, i + 1, i + args.cols + 1, i + args.cols]].astype(np.int32)
        m = np.zeros_like(gray); cv2.fillConvexPoly(m, q, 255)
        return cv2.mean(gray, m)[0]
    last = (args.rows - 2) * args.cols + (args.cols - 2)   # 반대쪽 끝 칸
    return c if quad_mean(0) <= quad_mean(last) else c[::-1].copy()


def collect(n, src):
    cam = LatestFrame(src, W, H)
    found, prev_seq, t0, size = [], None, time.time(), None
    print(f"cam{n} ({mask(src)}) 보드 찾는 중...")
    while len(found) < args.frames and time.time() - t0 < args.timeout:
        f, _, seq = cam.read()
        if f is None or seq == prev_seq:
            time.sleep(0.02); continue
        prev_seq = seq
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        c = detect(gray)
        if c is not None:
            found.append(canonical(gray, c)); size = gray.shape[::-1]
            print(f"  cam{n} {len(found)}/{args.frames}")
    cam.close()
    if not found:
        sys.exit(f"cam{n}: {args.timeout:.0f}초 안에 체커보드를 못 찾음. 보드를 카메라 쪽으로 옮기거나 더 크게 출력하세요")
    spread = float(np.std(np.stack(found), axis=0).max())
    return np.mean(np.stack(found), axis=0), size, len(found), spread


obj = np.zeros((args.rows * args.cols, 3), np.float64)
obj[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square
obj[:, 2] = args.board_height      # 보드 윗면 높이 → 바닥 = z 0

pts, sizes, info = {}, {}, {}
for n, src in enumerate(args.cams):
    pts[n], sizes[n], cnt, spread = collect(n, src)
    info[n] = {"detections": cnt, "corner_std_px": round(spread, 2)}


def solve(objp):
    out = {}
    for n in (0, 1):
        d = np.load(CALIB / f"cam{n}_intr.npz")
        K, dist = d["K"], d["dist"]
        ok, rvec, tvec = cv2.solvePnP(objp, pts[n], K, dist, flags=cv2.SOLVEPNP_IPPE)
        ok, rvec, tvec = cv2.solvePnP(objp, pts[n], K, dist, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - pts[n], axis=1)
        out[n] = dict(K=K, dist=dist, R=R, t=tvec, C=(-R.T @ tvec).ravel(), err=err)
    return out


sol = solve(obj)
if sol[0]["C"][2] < args.board_height:          # z 가 바닥 아래를 향하면 y 를 뒤집어 z = 위로 맞춘다
    obj[:, 1] *= -1
    sol = solve(obj)

for n in (0, 1):
    s = sol[n]
    np.savez(CALIB / f"cam{n}.npz", K=s["K"], dist=s["dist"], R=s["R"], t=s["t"], res=np.array(sizes[n]))

# 검증: 두 카메라로 코너를 삼각측량해 실제 보드 좌표와 비교 (계산에 쓴 점이지만 두 카메라 일관성을 본다)
c0, c1 = load_cam(str(CALIB / "cam0.npz")), load_cam(str(CALIB / "cam1.npz"))
u0, u1 = undistort(c0, pts[0]), undistort(c1, pts[1])
tri = np.array([triangulate(c0, c1, a, b)[0] for a, b in zip(u0, u1)])
tri_err_cm = np.linalg.norm(tri - obj, axis=1) * 100
baseline = float(np.linalg.norm(sol[0]["C"] - sol[1]["C"]))
ray = [sol[n]["R"].T @ np.array([0, 0, 1.0]) for n in (0, 1)]
angle = float(np.degrees(np.arccos(np.clip(np.dot(ray[0], ray[1]), -1, 1))))

x0, y0, x1, y1 = args.room
world = {"_설명": "10-2 자동판(calib_extrinsics_board.py): 원점=바닥 체커보드 첫 내부 코너, x=보드 가로, y=보드 세로, z=위. "
                  "아래 네 점은 proximity_app 의 방 범위(room_bounds)용이며 측정한 점이 아님",
         "points": [{"id": f"R{i}", "xyz": [x, y, 0.0], "check": True}
                    for i, (x, y) in enumerate([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], 1)]}
json.dump(world, open(CALIB / "world_points.json", "w"), ensure_ascii=False, indent=1)

row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": "10-2", "method": "floor-checkerboard",
       "board": [args.cols, args.rows, args.square], "board_height_m": args.board_height,
       "cams": {f"cam{n}": {"pos_m": sol[n]["C"].round(3).tolist(),
                            "reproj_mean_px": round(float(sol[n]["err"].mean()), 3),
                            "reproj_max_px": round(float(sol[n]["err"].max()), 3), **info[n]} for n in (0, 1)},
       "board_center_xy": obj[:, :2].mean(0).round(3).tolist(),
       "baseline_m": round(baseline, 3), "view_angle_deg": round(angle, 1),
       "triangulation_err_cm": {"mean": round(float(tri_err_cm.mean()), 2), "max": round(float(tri_err_cm.max()), 2)}}
with open(RESULTS / "10_extrinsics.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")

for n in (0, 1):
    print(f"cam{n} 위치(m) {sol[n]['C'].round(2)}  재투영 평균 {sol[n]['err'].mean():.2f}px")
print(f"보드 가운데 바닥 좌표(m) {obj[:, :2].mean(0).round(3).tolist()}  ← 타겟을 이 자리에 두면 --target-xy 로 그대로 쓴다")
print(f"두 카메라 거리 {baseline:.2f}m, 시선 사이 각 {angle:.0f}도")
print(f"삼각측량 검증 오차 평균 {tri_err_cm.mean():.1f}cm, 최대 {tri_err_cm.max():.1f}cm")
print("저장: calib/cam0.npz, calib/cam1.npz, calib/world_points.json, results/10_extrinsics.jsonl")
