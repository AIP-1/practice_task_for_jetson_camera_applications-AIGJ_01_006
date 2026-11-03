"""10단계 오프라인 분석 (새 측정 없음, 시뮬레이션)
카메라를 철거한 뒤, 저장소에 남은 보정 결과만으로 다음을 계산한다.
  1) 배치 5 구도에서 사람의 발목·어깨·코가 화면에 들어오는 거리 → 10-28 실측의 보인 비율과 대조
  2) ankle1 의 발목 높이 편향(발목을 바닥 z=0 으로 보는 효과)을 실제 앱 함수(ray_plane)로 계산 → +27cm 분해
  3) 카메라를 들었을 때(제안 구도) stereo 삼각측량의 이론 오차 (픽셀 잡음 몬테카를로)
  4) 제안 구도에서 보드를 얼마나 올려야 두 화면에 들어오는지 (--board-height 값)

입력: results/10_calib.jsonl(K), results/10_extrinsics.jsonl(배치 5 카메라 위치)
가정(실측 아님): 카메라 방향은 저장소에 없어서(R 은 2호기 npz 에만 있음) "광축이 보드 가운데를 향함"으로 둔다.
         렌즈 왜곡은 무시, 서 있는 사람 발목 0.08m · 어깨 1.40m · 코 1.60m, 사람은 타겟에서 카메라 반대쪽에 섬.
출력: results/10_offline_sim.json (사람이 읽는 표는 docs/02-results.md 10단계 "오프라인 분석")
실행: cd prox && python offline_10.py   (cv2·numpy 만 필요, 카메라·GPU 불필요)
"""
import json
from pathlib import Path
import numpy as np
import stereo as S

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
W, H = 640, 480
BODY = {"ankle": 0.08, "shoulder": 1.40, "nose": 1.60}
DISTS = [0.5, 1.0, 2.0, 3.0]
SEEN_10_28 = {0.5: 1.0, 1.0: 1.0, 2.0: 0.04, 3.0: 0.0}   # results/10_proximity.jsonl 10-28 ankle1 seen_ratio


def jsonl(name):
    return [json.loads(l) for l in open(RES / name, encoding="utf-8")]


def used_K():
    """10_calib.jsonl 에서 '최종 사용' 으로 적힌 줄의 K (cam0 2차, cam1 3차)"""
    rows = jsonl("10_calib.jsonl")
    pick = {"cam0": [r for r in rows if r["name"] == "cam0" and "최종 사용" in r["note"] and r["res"] == [640, 480]],
            "cam1": [r for r in rows if r["name"] == "cam1" and "최종 사용" in r["note"]]}
    return {k: np.array(v[-1]["K"], float) for k, v in pick.items()}


def batch5():
    """배치 5(10-28) 위치 보정 결과: 카메라 위치, 타겟(보드 가운데)"""
    r = [r for r in jsonl("10_extrinsics.jsonl") if r["time"].startswith("2026-10-28")][-1]
    return {n: np.array(r["cams"][n]["pos_m"], float) for n in ("cam0", "cam1")}, np.array(r["board_center_xy"], float)


def make_cam(K, C, look_xy, pitch_deg=None):
    """위치 C 에서 look_xy(바닥) 방향을 보는 카메라. pitch_deg 를 주면 수평 방향만 맞추고 그 각도로 숙인다(음수 = 아래)"""
    look = np.array([look_xy[0], look_xy[1], 0.0])
    f = look - C
    if pitch_deg is not None:
        h = np.array([f[0], f[1], 0.0]) / np.linalg.norm(f[:2])
        p = np.radians(pitch_deg)
        f = h * np.cos(p) + np.array([0, 0, 1.0]) * np.sin(p)
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)          # 화면 오른쪽
    d = np.cross(f, r)                                              # 화면 아래쪽
    R = np.vstack([r, d, f])                                        # 월드 → 카메라 (x 오른쪽, y 아래, z 앞)
    t = (-R @ C).reshape(3, 1)
    return {"K": K, "dist": np.zeros(5), "R": R, "t": t, "P": K @ np.hstack([R, t]),
            "C": C, "Kinv": np.linalg.inv(K)}


def in_frame(cam, X):
    if (cam["R"] @ (np.asarray(X) - cam["C"]))[2] <= 0:
        return False
    u, v = S.project(cam, X)
    return bool(0 <= u < W and 0 <= v < H)


def person_xy(target, cam_xy, r):
    """타겟에서 카메라 반대쪽으로 r m 떨어진 바닥 점"""
    u = target - cam_xy
    return target + r * u / np.linalg.norm(u)


def pitch_of(cam):
    f = cam["R"][2]
    return float(np.degrees(np.arcsin(f[2])))


def visibility(cams, target):
    rows = []
    for r in DISTS:
        p = person_xy(target, cams["cam0"]["C"][:2], r)
        row = {"r_m": r, "cam_to_person_m": round(float(np.linalg.norm(p - cams["cam0"]["C"][:2])), 2)}
        for n, c in cams.items():
            row[n] = {k: in_frame(c, [*p, z]) for k, z in BODY.items()}
        rows.append(row)
    return rows


def ankle_bias(cam, target):
    """실제 발목(높이 0.08)을 화면에 투영 → 앱처럼 z=0 평면으로 되돌림 → 타겟까지 거리 차이"""
    out = []
    for r in [0.5, 1.0]:
        p = person_xy(target, cam["C"][:2], r)
        X = S.ray_plane(cam, S.project(cam, [*p, BODY["ankle"]]), 0.0)
        d = float(np.linalg.norm(X[:2] - target))
        out.append({"r_m": r, "app_d_m": round(d, 3), "bias_m": round(d - r, 3)})
    return out


def stereo_error(cams, target, sigma_px, n=500, seed=0):
    """어깨 픽셀에 표준편차 sigma_px 잡음을 넣고 삼각측량 → 타겟까지 거리 오차 (중앙값·95%)"""
    rng = np.random.default_rng(seed)
    out = []
    for r in DISTS:
        p = person_xy(target, cams["cam0"]["C"][:2], r)
        X = np.array([*p, BODY["shoulder"]])
        if not all(in_frame(c, X) for c in cams.values()):
            out.append({"r_m": r, "visible": False}); continue
        p0, p1 = S.project(cams["cam0"], X), S.project(cams["cam1"], X)
        errs = []
        for _ in range(n):
            Xh, _ = S.triangulate(cams["cam0"], cams["cam1"], p0 + rng.normal(0, sigma_px, 2), p1 + rng.normal(0, sigma_px, 2))
            errs.append(abs(np.linalg.norm(Xh[:2] - target) - r))
        out.append({"r_m": r, "visible": True, "median_cm": round(float(np.median(errs)) * 100, 1),
                    "p95_cm": round(float(np.percentile(errs, 95)) * 100, 1)})
    return out


def board_min_height(cams, target, board=(0.16, 0.10)):
    """제안 구도에서 보드(기본 9x6 2cm 내부 코너 폭 16x10cm, 큰 보드 7x4 9cm 는 54x27cm)의 네 모서리가 두 화면에 모두 들어오는 최소 높이와 그때 화면 폭"""
    bx, by = board[0] / 2, board[1] / 2
    for h in np.arange(0.0, 1.21, 0.05):
        corners = [[target[0] + sx * bx, target[1] + sy * by, h] for sx in (-1, 1) for sy in (-1, 1)]
        if all(in_frame(c, X) for c in cams.values() for X in corners):
            width = {n: round(float(np.linalg.norm(S.project(c, corners[0]) - S.project(c, corners[2]))), 1)
                     for n, c in cams.items()}
            return {"height_m": round(float(h), 2), "board_width_px": width}
    return None


def main():
    K, (pos, target) = used_K(), batch5()
    actual = {n: make_cam(K[n], pos[n], target) for n in pos}
    out = {"kind": "offline simulation (no new measurement)",
           "assumption": "광축이 보드 가운데를 향함, 왜곡 무시, 발목 0.08·어깨 1.40·코 1.60 m, 사람은 타겟에서 카메라 반대쪽",
           "batch5": {n: {"pos_m": pos[n].tolist(), "pitch_deg": round(pitch_of(c), 1)} for n, c in actual.items()},
           "visibility_batch5": visibility(actual, target),
           "seen_ratio_10_28": SEEN_10_28,
           "ankle_bias_batch5": ankle_bias(actual["cam0"], target)}
    # 숙인 각도 가정이 몇 도 틀려도 결론(어깨 안 보임)이 유지되는지
    out["sensitivity"] = []
    for dp in (-5, 5):
        c = {n: make_cam(K[n], pos[n], target, pitch_of(actual[n]) + dp) for n in pos}
        out["sensitivity"].append({"pitch_shift_deg": dp, "visibility": visibility(c, target)})
    # 제안 구도: 같은 위치에서 카메라를 들어(-12도) 0.5~3m 사람의 발목~어깨가 보이게
    prop = {n: make_cam(K[n], pos[n], target, -12.0) for n in pos}
    out["proposed"] = {"pitch_deg": -12.0, "visibility": visibility(prop, target),
                       "stereo_err_sigma2px": stereo_error(prop, target, 2.0),
                       "stereo_err_sigma5px": stereo_error(prop, target, 5.0),
                       "board_min_height_9x6_2cm": board_min_height(prop, target),
                       "board_min_height_7x4_9cm": board_min_height(prop, target, board=(0.54, 0.27))}
    (RES / "10_offline_sim.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
