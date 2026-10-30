# -*- coding: utf-8 -*-
"""Run fl.py jobs from a text file (one 'dataset tag key=value ...' per line) with bounded GPU concurrency.
Jobs whose result JSON already exists are skipped.  python run_queue.py jobs.txt [concurrency]"""
import sys, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent; RUNS = ROOT / "runs"; LOGS = ROOT / "logs"; LOGS.mkdir(exist_ok=True)
jobs = [l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
todo = [j for j in jobs if not (RUNS / f"{j[0]}_{j[1]}.json").exists()]
print(f"{len(todo)} of {len(jobs)} jobs to run, concurrency {N}", flush=True)
running = []
while todo or running:
    while todo and len(running) < N:
        j = todo.pop(0)
        f = open(LOGS / f"{j[0]}_{j[1]}.log", "w")
        running.append((j, subprocess.Popen([sys.executable, "fl.py", *j], stdout=f, stderr=subprocess.STDOUT, cwd=Path(__file__).parent), f))
    for item in list(running):
        j, p, f = item
        if p.poll() is not None:
            f.close(); running.remove(item)
            last = open(LOGS / f"{j[0]}_{j[1]}.log").read().strip().splitlines()[-1:]
            print(f"[{time.strftime('%H:%M:%S')}] exit {p.returncode} {' '.join(j[:2])}: {last[0] if last else ''}", flush=True)
    time.sleep(2)
print("ALL DONE", flush=True)
