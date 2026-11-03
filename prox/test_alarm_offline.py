"""10-6 접근 알람 로직 오프라인 검증 (새 측정 없음, 합성 거리 신호)
proximity_app.py 의 ProximityAlarm 클래스 소스를 그대로 꺼내 실행한다(앱은 카메라·YOLO 를 열어서 import 할 수 없음).
강의자료 10-6 시나리오 5개를 거리 시계열로 만들어 각 3회(시드 0~2) 돌리고, 기대 알람이 나오는지 센다.
  - 10 Hz, 걷는 속도 1.0 m/s, 거리 잡음 σ 0.02 m (10-28 ankle1 실측 d_std 0.009~0.018 m 보다 조금 크게)
  - "+0.27 m 편향" 변형: 10-28 ankle1 의 일정한 오차를 더했을 때 근접 알람이 실제 몇 m 에서 울리는지
  - 경계 잡음 실험: 1.0 m 경계에 10초 서 있을 때 잡음 σ 를 키우며 알람 깜빡임 수
한계: 사람 검출·추적·자세(앉음) 판정은 포함하지 않는다. 실제 카메라 입력으로 확인한 것이 아니다.
실행: cd prox && python test_alarm_offline.py   (numpy 만 필요) → results/10_alarm_offline.jsonl
"""
import ast, json, sys, types
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "10_alarm_offline.jsonl"
NEAR, FAR, HZ, V = 1.0, 3.0, 10, 1.0


def load_alarm():
    """proximity_app.py 에서 ProximityAlarm 정의만 꺼내 같은 코드로 실행"""
    src = (HERE / "proximity_app.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "ProximityAlarm")
    ns = {"args": types.SimpleNamespace(ttc=2.0), "CLOSE_V": 0.5}
    exec(compile(ast.Module([node], []), "proximity_app.py", "exec"), ns)
    return ns["ProximityAlarm"]


def walk(d0, d1, hold0=1.0, hold1=1.0):
    """d0 에 hold0 초 서 있다가 V m/s 로 d1 까지 걷고 hold1 초 서 있는 거리 시계열"""
    n = int(abs(d1 - d0) / V * HZ)
    return [d0] * int(hold0 * HZ) + list(np.linspace(d0, d1, n)) + [d1] * int(hold1 * HZ)


def pass_by(lateral=2.0, half=4.0):
    """타겟 옆 lateral m 를 직선으로 지나감"""
    xs = np.arange(-half, half, V / HZ)
    return list(np.hypot(xs, lateral))


def run(Alarm, true_d, sigma, seed, bias=0.0):
    rng = np.random.default_rng(seed)
    a, events = Alarm(NEAR, FAR), []
    for i, d in enumerate(true_d):
        meas = max(0.0, d + bias + rng.normal(0, sigma))
        for e in a.update(i / HZ, meas):
            events.append({"t": round(i / HZ, 1), "true_d": round(float(d), 2), "event": e})
    return events


def count(events, prefix):
    return sum(e["event"].startswith(prefix) for e in events)


SCENARIOS = [
    # 이름, 거리 시계열, 기대 판정 함수, 자료의 기대 결과
    ("4m에서 타겟으로 걸어옴", walk(4.0, 0.5), lambda ev: count(ev, "접근 시작") >= 1 and count(ev, "알람: 근접") == 1,
     "접근 시작 → 알람: 근접"),
    ("타겟 앞 0.5m에서 4m 밖으로 나감", walk(0.5, 4.5), lambda ev: count(ev, "알람: 이탈") == 1, "알람: 이탈"),
    ("1.0m 경계에서 10초 서 있음", [NEAR] * (10 * HZ), lambda ev: count(ev, "알람: 근접") <= 1, "알람 1회만 (깜빡임 없음)"),
    ("타겟 2m 옆을 지나감", pass_by(2.0), lambda ev: count(ev, "알람: 근접") == 0, "근접 알람 없음"),
    ("타겟 옆 책상에 앉음 (0.6m)", walk(2.0, 0.6, hold1=5.0), lambda ev: count(ev, "알람: 근접") == 1,
     "근접 알람 (자세 = 앉음은 stereo 높이가 필요해 여기서는 확인 못 함)"),
]


def main():
    Alarm = load_alarm()
    rows, ok_all = [], True
    print("| 시나리오 (3회씩) | 기대 결과 | 맞은 횟수 | 비고 |\n| --- | --- | --- | --- |")
    for name, d, check, expect in SCENARIOS:
        runs = [run(Alarm, d, 0.02, s) for s in range(3)]
        hits = sum(check(ev) for ev in runs)
        ok_all &= hits == 3
        note = "; ".join(f"{e['event']}@{e['true_d']}m" for e in runs[0]) or "알람 없음"
        print(f"| {name} | {expect} | {hits}/3 | 1회차: {note} |")
        rows.append({"step": "10-6-offline", "scenario": name, "expect": expect, "hits": hits, "runs": 3,
                     "sigma_m": 0.02, "events_run0": runs[0]})

    # ankle1 편향(+0.27m)을 더하면 근접 알람이 실제 몇 m 에서 울리나
    ev = run(Alarm, walk(4.0, 0.3), 0.02, 0, bias=0.27)
    near_true = [e["true_d"] for e in ev if e["event"].startswith("알람: 근접")]
    print(f"\n+0.27m 편향: 근접 알람이 실제 거리 {near_true} m 에서 울림 (기준 {NEAR} m)")
    rows.append({"step": "10-6-offline", "scenario": "ankle1 +0.27m 편향, 4m→0.3m", "near_alarm_true_d_m": near_true})

    # 경계 잡음: σ 를 키우면 1.0m 경계에서 알람이 몇 번 울리나 (히스테리시스 폭 0.2m)
    print("\n| 거리 잡음 σ (m) | 1.0m 경계 10초 근접 알람 수 (3회) |\n| --- | --- |")
    for sigma in (0.02, 0.05, 0.10, 0.20):
        n = [count(run(Alarm, [NEAR] * (10 * HZ), sigma, s), "알람: 근접") for s in range(3)]
        print(f"| {sigma} | {n} |")
        rows.append({"step": "10-6-offline", "scenario": "1.0m 경계 잡음", "sigma_m": sigma, "near_alarms": n})

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"\n기록: {OUT.relative_to(HERE.parent)}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
