"""10-2 카메라 위치 보정: 웹에서 두 카메라 스냅샷 위의 기준점을 클릭 → solvePnP로 카메라 위치·방향 계산
기준점 실제 좌표는 calib/world_points.json에 미리 적어 둔다 ("check": true 인 점은 계산에 쓰지 않고 검증에만 사용)
사용법: python calib_extrinsics.py --cams rtsp://... http://<1호기>:8140/stream.mjpg  →  브라우저에서 http://<2호기>:8000
자료와 다른 점: --cams 에 장치 번호 대신 영상 주소를 받는다. 보정 파일은 순서대로 cam0, cam1 이다.
"""
import argparse, json, os, time
import cv2, numpy as np, uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, Response
from stereo import load_world, undistort, triangulate, load_cam
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lab"))
from camera import LatestFrame, mask   # RTSP(2호기 VIGI)·USB·HTTP MJPEG(1호기 cam_server.py)를 같은 방식으로 연다

ap = argparse.ArgumentParser()
ap.add_argument("--cams", nargs=2, required=True, help="cam0, cam1 영상 주소 (RTSP·HTTP 또는 USB 번호)")
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
WORLD = load_world("calib/world_points.json")
WMAP = {p["id"]: p for p in WORLD}
CLICKS_F = "calib/clicks.json"
clicks = json.load(open(CLICKS_F)) if os.path.exists(CLICKS_F) else {"0": {}, "1": {}}
snaps = {}

def take_snaps():
    for n, src in enumerate(args.cams):
        cam = LatestFrame(src, W, H)
        time.sleep(1.5)              # 자동 노출 안정화 + 첫 키프레임 대기
        f, _, _ = cam.read()
        cam.close()
        assert f is not None, f"cam{n} 영상을 받지 못함: {mask(src)}"
        snaps[str(n)] = f
        cv2.imwrite(f"calib/snap{n}.jpg", f)

def intr(n):
    d = np.load(f"calib/cam{n}_intr.npz")
    return d["K"], d["dist"]

def solve():
    out = {}
    for n in ("0", "1"):
        K, dist = intr(int(n))
        ids = [i for i in clicks[n] if i in WMAP and not WMAP[i].get("check")]
        if len(ids) < 6:
            out[n] = {"error": f"계산용 기준점 {len(ids)}개 (6개 이상 필요)"}; continue
        obj = np.float64([WMAP[i]["xyz"] for i in ids])
        img = np.float64([clicks[n][i] for i in ids])
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        errs = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        R, _ = cv2.Rodrigues(rvec)
        h, w = snaps[n].shape[:2]
        np.savez(f"calib/cam{n}.npz", K=K, dist=dist, R=R, t=tvec, res=np.array([w, h]))
        C = (-R.T @ tvec).ravel()
        out[n] = {"pos": C.round(2).tolist(), "reproj": {i: round(float(e), 1) for i, e in zip(ids, errs)},
                  "mean_px": round(float(errs.mean()), 2)}
    # 두 카메라 모두 클릭한 점으로 3D 검증 (check 점 포함)
    if "error" not in out.get("0", {"error": 1}) and "error" not in out.get("1", {"error": 1}):
        c0, c1 = load_cam("calib/cam0.npz"), load_cam("calib/cam1.npz")
        ver = {}
        for i in set(clicks["0"]) & set(clicks["1"]) & set(WMAP):
            p0, p1 = undistort(c0, [clicks["0"][i]])[0], undistort(c1, [clicks["1"][i]])[0]
            X, _ = triangulate(c0, c1, p0, p1)
            ver[i] = {"check": bool(WMAP[i].get("check")),
                      "err_cm": round(float(np.linalg.norm(X - np.array(WMAP[i]["xyz"]))) * 100, 1)}
        out["verify"] = ver
    return out

app = FastAPI()

@app.get("/img/{n}.jpg")
def img(n: str):
    f = snaps[n].copy()
    for i, (u, v) in clicks[n].items():
        col = (0, 200, 255) if WMAP.get(i, {}).get("check") else (0, 255, 0)
        cv2.circle(f, (int(u), int(v)), 5, col, -1)
        cv2.putText(f, i, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return Response(cv2.imencode(".jpg", f)[1].tobytes(), media_type="image/jpeg")

@app.post("/click")
def click(d: dict = Body(...)):
    clicks[str(d["cam"])][d["id"]] = [float(d["u"]), float(d["v"])]
    json.dump(clicks, open(CLICKS_F, "w"), indent=1)
    return {"ok": True}

@app.post("/undo")
def undo(d: dict = Body(...)):
    clicks[str(d["cam"])].pop(d["id"], None)
    json.dump(clicks, open(CLICKS_F, "w"), indent=1)
    return {"ok": True}

@app.post("/snap")
def snap():
    take_snaps(); return {"ok": True}

@app.post("/solve")
def do_solve():
    return solve()

@app.get("/points")
def points():
    return {"points": WORLD, "clicks": clicks}

PAGE = """<!doctype html><meta charset=utf-8><title>카메라 위치 보정</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}.row{display:flex;gap:12px;flex-wrap:wrap}
.row div{flex:1;min-width:300px}img{width:100%;cursor:crosshair;border:1px solid #444}select,button{font-size:15px;padding:4px 10px}
pre{background:#222;padding:8px;white-space:pre-wrap}</style>
<h2>카메라 위치 보정</h2>
<p>기준점 선택 → 해당 카메라 이미지에서 그 점을 클릭. 초록 = 계산용, 주황 = 검증용(check). 카메라마다 계산용 6개 이상.</p>
<p><select id=pid></select> <button onclick="undo(0)">cam0 선택점 지우기</button> <button onclick="undo(1)">cam1 선택점 지우기</button>
<button onclick="snapNew()">새 스냅샷</button> <button onclick="solve()">계산</button></p>
<div class=row><div>cam0<br><img id=i0 src="/img/0.jpg" onclick="clk(event,0)"></div>
<div>cam1<br><img id=i1 src="/img/1.jpg" onclick="clk(event,1)"></div></div>
<h3>결과</h3><pre id=out>-</pre>
<script>
async function load(){const r=await (await fetch('/points')).json();
 document.getElementById('pid').innerHTML=r.points.map(p=>`<option value=${p.id}>${p.id} (${p.xyz.join(', ')})${p.check?' 검증':''}</option>`).join('');}
function refresh(){for(const n of [0,1])document.getElementById('i'+n).src=`/img/${n}.jpg?`+Date.now();}
async function clk(e,n){const im=e.target,u=e.offsetX*im.naturalWidth/im.clientWidth,v=e.offsetY*im.naturalHeight/im.clientHeight;
 await fetch('/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cam:n,id:pid.value,u,v})});refresh();}
async function undo(n){await fetch('/undo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cam:n,id:pid.value})});refresh();}
async function snapNew(){await fetch('/snap',{method:'POST'});refresh();}
async function solve(){const r=await (await fetch('/solve',{method:'POST'})).json();document.getElementById('out').textContent=JSON.stringify(r,null,1);}
load();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    take_snaps()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
