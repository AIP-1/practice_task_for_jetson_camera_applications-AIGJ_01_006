"""10단계 보조: 1호기의 USB 카메라 영상을 Tailscale 로 내보내는 MJPEG 서버.

10단계는 한 장비가 두 카메라 영상을 동시에 받아야 삼각측량을 할 수 있다. 우리 팀은 USB 카메라가 1호기,
IP 카메라(VIGI)가 2호기에 있어서, 1호기가 USB 영상을 HTTP MJPEG 로 내보내고 2호기가 RTSP 와 함께 받는다.
모델을 올리지 않으므로 1호기 GPU·메모리를 거의 쓰지 않는다. 프레임은 저장하지 않는다.

사용 (1호기에서): python cam_server.py --cam 0 --port 8140
받는 쪽 (2호기): http://<1호기 Tailscale 주소>:8140/stream.mjpg
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lab"))
from camera import LatestFrame  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cam", default="0", help="USB 장치 번호 (v4l2-ctl --list-devices 로 확인)")
ap.add_argument("--res", default="640x480")
ap.add_argument("--fps", type=float, default=15, help="내보내는 최대 fps (Tailscale 대역폭 절약)")
ap.add_argument("--quality", type=int, default=80)
ap.add_argument("--port", type=int, default=8140, help="1호기 8000·8086·8090·8091·8100·8130 은 이미 사용 중 (10-27 ss -ltn 확인)")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
cam = LatestFrame(args.cam, W, H)
app = FastAPI()


@app.get("/stream.mjpg")
def stream():
    def gen():
        last = -1
        while True:
            f, _, seq = cam.read()
            if f is None or seq == last:
                time.sleep(0.01)
                continue
            last = seq
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(1 / args.fps)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/health")
def health():
    f, ts, seq = cam.read()
    return {"ok": f is not None, "frames": seq, "age_s": round(time.time() - ts, 2) if ts else None,
            "size": None if f is None else [f.shape[1], f.shape[0]]}


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
    finally:
        cam.close()
