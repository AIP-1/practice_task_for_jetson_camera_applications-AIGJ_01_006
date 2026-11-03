"""카메라 입력 공통 모듈: USB 웹캠(번호)과 IP 카메라(RTSP 주소)를 같은 방식으로 연다.

자료 8~10단계는 cv2.VideoCapture(번호, CAP_V4L2) 로 USB 웹캠만 가정한다.
2호기는 IP 카메라라서, 같은 코드가 RTSP 도 받도록 입력을 이 모듈로 모은다.

RTSP 는 OpenCV 내부 버퍼에 프레임이 쌓여 처리보다 느리면 지연이 수 초까지 늘어난다.
그래서 별도 스레드가 계속 읽어 가장 최신 프레임 한 장만 들고 있게 한다.
"""
import os
import threading
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent


def _load_env():
    """저장소 루트의 .env 에서 CAM_URL 등을 읽는다 (python-dotenv 의존을 피하려고 직접 파싱)."""
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def default_source():
    """CAM_URL 이 있으면 RTSP, 없으면 USB 0번."""
    _load_env()
    return os.environ.get("CAM_URL", "0")


def mask(source):
    """로그에 RTSP 계정이 찍히지 않게 가린다.

    계정과 주소는 마지막 "@" 로 나눈다. 사용자 이름이 이메일처럼 "@" 를 품고 있으면
    첫 "@" 로 나눌 때 비밀번호가 그대로 찍혔다 (README 문제 해결 기록 6).
    """
    s = str(source)
    if "@" in s and "://" in s:
        scheme, rest = s.split("://", 1)
        return f"{scheme}://***@{rest.rsplit('@', 1)[1]}"
    return s


def source_label(source):
    """결과 파일(공개 저장소)에 남길 입력 이름. 계정과 카메라 IP 를 빼고 스트림 경로만 남긴다.

    예) rtsp://admin:pw@10.0.0.5:554/stream2 -> rtsp://<카메라>/stream2, USB 0 -> usb:0
    """
    s = str(source)
    if "://" not in s:
        return f"usb:{s}"
    scheme, rest = s.split("://", 1)
    path = rest.rsplit("@", 1)[-1].split("/", 1)
    return f"{scheme}://<카메라>/{path[1] if len(path) > 1 else ''}"


def open_capture(source, width=None, height=None):
    s = str(source)
    if s.isdigit():
        cap = cv2.VideoCapture(int(s), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        # UDP 는 유실 시 화면이 깨지므로 TCP 로 받는다 (자료 4단계 ffmpeg 옵션과 같음)
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(s, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없음: {mask(s)}")
    return cap


class LatestFrame:
    """백그라운드에서 계속 읽고 최신 프레임만 보관한다. RTSP 가 끊기면 다시 연결한다."""

    def __init__(self, source=None, width=None, height=None, reconnect_s=2.0):
        self.source = default_source() if source is None else source
        self.size = (width, height)
        self.reconnect_s = reconnect_s
        self._frame, self._ts, self._seq = None, 0.0, 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = open_capture(self.source, width, height)
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                self._cap.release()
                time.sleep(self.reconnect_s)
                try:
                    self._cap = open_capture(self.source, *self.size)
                except RuntimeError:
                    pass
                continue
            with self._lock:
                self._frame, self._ts = frame, time.time()
                self._seq += 1

    def read(self):
        """(프레임 복사본, 받은 시각, 순번). 아직 없으면 (None, 0, 0)."""
        with self._lock:
            f = None if self._frame is None else self._frame.copy()
            return f, self._ts, self._seq

    def close(self):
        self._stop.set()
        self._t.join(timeout=2)
        self._cap.release()


if __name__ == "__main__":
    # 연결 확인: python lab/camera.py → 5초 동안 받은 fps 와 해상도 출력
    cam = LatestFrame()
    print("소스:", mask(cam.source))
    time.sleep(5)
    _, _, n = cam.read()
    f, _, _ = cam.read()
    print(f"5초 동안 {n}프레임 ({n / 5:.1f} fps), 해상도 {None if f is None else f.shape[1::-1]}")
    cam.close()
