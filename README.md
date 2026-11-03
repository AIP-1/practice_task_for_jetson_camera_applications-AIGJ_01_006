# Jetson 카메라 응용 실습 (EmbeddingGemma 2 8~10단계)

Jetson Orin Nano Super 8GB 에 USB 웹캠을 붙여 실시간 물체 검색(8단계), 사람 동작 인식(9단계), 카메라 2대 접근 감지(10단계)를 만든 과제 저장소다. 강의자료 "Jetson 실습 - EmbeddingGemma 2 멀티모달 임베딩" 의 카메라 응용 갈래이고, 앞부분(1~7단계)은 `embedding_gemma2-AIGJ_01_006` 저장소에 따로 냈다. 8~10단계도 1~2단계 설치와 `lab/eg2.py` 임베딩 로더를 쓰므로 코드 전체가 함께 들어 있다.

## 요약 (문제 → 접근 → 결과)

- **문제**: 강의실 카메라 한두 대와 보드 하나로 "무엇이 있나 / 무슨 동작을 하나 / 사람이 얼마나 가까운가" 를 숫자로 바꿀 수 있는가. 검출기가 모르는 물체와 프레임마다 흔들리는 판정을 어떻게 다루는가.
- **접근**: 같은 물체·같은 동작 대본으로 방식끼리 통제 비교(8단계 yolo / relabel / yoloe / yoloe+relabel, 9단계 raw / smooth / GRU). 10단계는 렌즈·위치 보정 후 사람이 줄자로 잰 자리에 서서 거리 오차를 쟀다.
- **결과** (MAXN_SUPER, bfloat16, Logitech C920 1280x720)

| 단계 | 핵심 결과 | 근거 |
|---|---|---|
| 8 물체 검색 | 라벨 목록 밖 물체는 가장 비슷한 이름으로 억지로 붙는다(은색 캔 → `컵`, 슬림 캔 → `cell phone 0.84`). **YOLOE 박스 + 임베딩 재라벨**이 캔 8/8·휴대폰 8/8 정답, 검색 1위 전부 정답. 검출 약 12 fps, 크롭당 임베딩 0.75 s, 보드 RAM 약 5.6 GB | [reports/step8.md](reports/step8.md) |
| 9 동작 인식 | 프레임 단위(raw) 판정은 실제 약 3회 동작을 17~48회로 셌고 손 흔들기는 0회. 0.6 s 다수결(smooth)로 1/3~1/4 로 줄고 손 흔들기 5회 검출. GRU 는 같은 녹화 검증 100% 였지만 처음 보는 위치 25%, 3 m 실시간 14% | [reports/step9.md](reports/step9.md) |
| 10 두 카메라 접근 감지 (일부) | 렌즈 보정 RMS 0.063·1.81 px, 위치 보정 재현성 0.1 cm 이하. 한 카메라 발목 방식(ankle1)으로 0.5·1.0 m → 0.76·1.28 m (약 +27 cm 일정한 오차, 25.5 fps). 두 카메라 삼각측량(stereo·fused)과 실기 접근 알람은 카메라 구도 문제로 측정 못 함, 알람 로직은 합성 신호 15/15 | [reports/step10.md](reports/step10.md) |

가장 크게 결과를 바꾼 설계 변수: **판정 시간 창**. 같은 모델·같은 영상에서 0.6 s 다수결만 넣어도 9단계 카운트 오차가 3~4배 줄었다. 8단계에서는 모델을 바꾸는 것보다 **라벨 결정을 검출기 대신 임베딩에 맡기는 것**이 오인식을 더 크게 줄였다.

## 역할

| 단계 | 담당 | 비고 |
|---|---|---|
| 1~6 (환경, 임베딩 기초, 자원 측정) | 나민서 (iamminseo) | 8~10단계가 쓰는 설치·로더 |
| 8 웹캠 물체 검색, 9 동작 인식 | 강성찬 | `lab/app.py`, `lab/pose_app.py`, `lab/train_pose.py` |
| 10 두 카메라 접근 감지 | 나민서 | |

## 환경

- Jetson Orin Nano Engineering Reference Developer Kit Super, RAM 8GB, NVMe
- JetPack 7.2.1 (L4T R39.2.1), Ubuntu 24.04, Python 3.12, CUDA 13.2, 전력 모드 MAXN_SUPER, `jetson_clocks` 미사용
- torch 2.14.1+cu132, sentence-transformers 6.2.0, ultralytics 8.4.184
- USB 웹캠 Logitech C920 (`/dev/video0`), 10단계는 1·2호기에 1대씩

## 설치

```bash
python3 -m venv ~/eg2 && source ~/eg2/bin/activate
pip install -U pip wheel
python -m pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/cu132
pip freeze | grep -E '^(torch|torchvision)==' > ~/eg2/torch-pin.txt
pip install -c ~/eg2/torch-pin.txt -U \
  "sentence-transformers[image,audio,video]>=6.1.0" transformers accelerate \
  soundfile librosa av pillow huggingface_hub psutil ultralytics fastapi uvicorn
pip uninstall -y torchaudio
hf download google/embeddinggemma-2
```

## 실행

공용 보드라서 모델을 올리기 전에 `free -h` 로 available 이 2.5GB 이상인지 확인한다.

```bash
cd lab
python app.py --mode yoloe --relabel   # 8단계: 웹캠 물체 검색, http://<Jetson IP>:8200
python pose_app.py --raw               # 9단계: 프레임 단위 판정
python pose_app.py                     # 9단계: 0.6초 다수결 + 손 흔들기 1.5초 판정
```

10단계 (`prox/` 에서. cam0 = 2호기 USB, cam1 = 1호기 USB):

```bash
python cam_server.py --cam 0 --port 8140                       # 1호기에서: 영상 MJPEG 송출
python cam_server.py --cam 0 --port 8141                       # 2호기에서
python calib_intrinsics.py --name cam0 --src http://127.0.0.1:8141/stream.mjpg --square 0.02
python calib_extrinsics_board.py --cams http://127.0.0.1:8141/stream.mjpg http://<1호기 Tailscale 주소>:8140/stream.mjpg --square 0.02 --board-height 0
python proximity_app.py --cams http://127.0.0.1:8141/stream.mjpg http://<1호기 Tailscale 주소>:8140/stream.mjpg --mode ankle1 --target-xy 0.08,-0.05
python measure_10_7.py --label "1.0m, 서 있음" --true-d 1.0 --seconds 10
python offline_10.py && python test_alarm_offline.py          # 카메라 없이: 보정값 기하 계산, 알람 로직 검증
```

## 실패 → 원인 → 해결

- **8단계 슬림 캔이 휴대폰으로 튐**: 얼굴 옆에 세워 든 슬림 캔을 YOLOE 가 프레임마다 `cell phone 0.84` 로 봤다. 같은 크롭의 임베딩은 `캔 음료` 0.67~0.73 으로 꾸준했다 → 흔들리는 건 검출기라고 보고 `--relabel` 을 yoloe 에도 붙임 → 캔 8/8.
- **9단계 카운트 부풂**: raw 는 라벨이 깜빡일 때마다 새 동작으로 셌다 → 다수결 창 도입. 그래도 손 흔드는 동안 반대 손이 올라가 한팔들기·만세가 섞임 → 규칙 임계값 한계로 기록, GRU 시도했으나 처음 보는 위치에서 25% 로 일반화 실패.
- **10단계 stereo 측정 불가**: 위치 보정을 위해 카메라를 바닥 보드 쪽으로 숙였더니 2 m 이상에서 사람 발목이 화면 밖 → 보정 구도와 측정 구도가 충돌. 보드를 의자 위에 올리고 `--board-height` 로 보정하는 재측정 절차를 남김.

## AI 활용과 검증

Claude 가 명령·스크립트를 단계별로 제안하고, 팀원이 Jetson 에서 직접 실행해 출력으로 검증했다. 보고서의 숫자는 모두 실제 실행 출력·tegrastats 로그에서 옮겼다. 10단계 오프라인 계산은 시뮬레이션이라 실측 칸으로 세지 않았다.

영상·모델 파일은 git 에 올리지 않는다 (`.gitignore`). 사람 얼굴이 나온 사진은 넣지 않았다.
