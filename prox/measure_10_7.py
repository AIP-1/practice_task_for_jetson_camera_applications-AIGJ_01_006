"""10-7 결과 비교 측정: 켜져 있는 proximity_app.py 의 /state 를 정해진 시간 동안 읽어 한 줄로 기록
사람이 테이프 지점(타겟에서 줄자로 잰 거리)에 서 있는 동안 실행한다. 사람 이미지는 저장하지 않는다.
사용법: python measure_10_7.py --label "1.0m, 서 있음" --true-d 1.0 --seconds 10
결과: results/10_proximity.jsonl (모드, 화면 거리 중앙값·표준편차, 보인 비율, 추정 방식, fps, 카메라 시간 차, RAM)
"""
import argparse, json, statistics, subprocess, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True, help='자료 10-7 표의 행 이름. 예: "2.0m, 서 있음"')
ap.add_argument("--true-d", type=float, default=None, help="줄자로 잰 실제 거리(m). 없으면 비움")
ap.add_argument("--seconds", type=float, default=10)
ap.add_argument("--url", default="http://127.0.0.1:8000/state")
args = ap.parse_args()


def mem_used_mb():
    info = dict(l.split(":", 1) for l in open("/proc/meminfo"))
    kb = lambda k: int(info[k].split()[0])
    return round((kb("MemTotal") - kb("MemAvailable")) / 1024)


def app_rss_mb():
    """proximity_app.py 프로세스 RSS (없으면 None)
    pgrep -f 는 tmux 를 띄운 셸(bash -c "... proximity_app.py ...")도 잡는다. 10-27 에는 첫 PID 가 그 셸로 추정되며
    3MB 가 기록됐다 → 잡힌 프로세스 중 RSS 가 가장 큰 것(파이썬 본체)을 쓴다."""
    rss = []
    try:
        pids = subprocess.check_output(["pgrep", "-f", "proximity_app.py"], text=True).split()
    except Exception:
        return None
    for pid in pids:
        try:
            for l in open(f"/proc/{pid}/status"):
                if l.startswith("VmRSS:"):
                    rss.append(int(l.split()[1]))
        except OSError:
            continue
    return round(max(rss) / 1024) if rss else None


ds, kinds, fps, skew, seen, polls, mode, ids = [], {}, [], [], 0, 0, None, set()
t_end = time.time() + args.seconds
while time.time() < t_end:
    try:
        s = json.load(urllib.request.urlopen(args.url, timeout=2))
    except Exception as e:
        print("상태를 읽지 못함:", e); time.sleep(0.5); continue
    polls += 1
    st = s.get("stats") or {}
    mode = st.get("mode", mode)
    if st.get("fps"):
        fps.append(st["fps"]); skew.append(st.get("skew_ms", 0))
    people = [p for p in s.get("people", []) if p.get("d") is not None]
    if people:
        seen += 1
        p = min(people, key=lambda q: q["d"])     # 측정 대상은 타겟에 가장 가까운 사람 한 명
        ds.append(p["d"]); kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1; ids.add(p["id"])
    time.sleep(0.2)

med = round(statistics.median(ds), 2) if ds else None
row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": "10-7", "mode": mode, "label": args.label,
       "true_d_m": args.true_d, "d_median_m": med,
       "d_std_m": round(statistics.pstdev(ds), 3) if len(ds) > 1 else None,
       "err_m": round(med - args.true_d, 2) if med is not None and args.true_d is not None else None,
       "seen_ratio": round(seen / polls, 2) if polls else None, "polls": polls, "kinds": kinds,
       "track_ids": len(ids), "fps_median": statistics.median(fps) if fps else None,
       "skew_ms_median": statistics.median(skew) if skew else None,
       "ram_used_mb": mem_used_mb(), "app_rss_mb": app_rss_mb(), "seconds": args.seconds}
out = Path(__file__).resolve().parent.parent / "results" / "10_proximity.jsonl"
out.parent.mkdir(exist_ok=True)
with open(out, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False))
print("화면 거리:", "없음" if med is None else f"{med} m", f"(보인 비율 {row['seen_ratio']})")
