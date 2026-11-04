"""성능 측정 보조 스크립트 (제공 코드 수정 아님, 측정 도구)
사용: cd ~/fastapi_resnet18 && python reports/measure.py <이미지경로>
조건: docker compose로 redis·postgres·web·worker 실행 중
"""
import csv, statistics, subprocess, sys, time
import httpx

URL = 'http://127.0.0.1:8000'
IMG = sys.argv[1]
N = 5
DATA = open(IMG, 'rb').read()
rows = []


def flush():  # Redis 캐시(DB 1) 비우기 → 다음 요청을 MISS로 만듦
    subprocess.run(['docker', 'compose', 'exec', '-T', 'redis', 'redis-cli', '-n', '1', 'FLUSHDB'],
                   check=True, capture_output=True)


def files():
    return {'image': ('img', DATA, 'application/octet-stream')}


def run_sync(c, i, cache):
    if cache == 'MISS':
        flush()
    t0 = time.perf_counter()
    r = c.post('/api/sync/classify/', files=files())
    ms = round((time.perf_counter() - t0) * 1000, 1)
    j = r.json()
    rows.append([i, 'sync', cache, r.status_code, ms, ms, j['inference_ms'], '', j['cache_hit'], j['top1']['label']])


def run_async(c, i, cache):
    if cache == 'MISS':
        flush()
    t0 = time.perf_counter()
    r = c.post('/api/async/classify/', files=files())
    http_ms = round((time.perf_counter() - t0) * 1000, 1)
    job = r.json()['job_id']
    while True:
        s = c.get(f'/api/async/jobs/{job}/').json()
        if s['status'] in ('done', 'failed'):
            break
        time.sleep(0.2)  # 폴링 간격 0.2초
    final_ms = round((time.perf_counter() - t0) * 1000, 1)
    label = s['top1']['label'] if s['top1'] else s['status']
    rows.append([i, 'async', cache, r.status_code, http_ms, final_ms, s['inference_ms'], job, s['cache_hit'], label])


with httpx.Client(base_url=URL, timeout=120) as c:
    # 워밍업 (기록 제외)
    flush(); c.post('/api/sync/classify/', files=files())
    flush(); run_async(c, 0, 'MISS'); rows.clear()
    for cache in ('MISS', 'HIT'):
        for i in range(1, N + 1):
            run_sync(c, i, cache)
    for cache in ('MISS', 'HIT'):
        for i in range(1, N + 1):
            run_async(c, i, cache)

with open('reports/data/timings.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run_id', 'path', 'cache_status', 'http_status', 'http_ms', 'final_ms',
                'inference_ms', 'job_id', 'cache_hit', 'top1_label'])
    w.writerows(rows)

print('run path  cache http  http_ms  final_ms  inference_ms  job  cache_hit  label')
for r in rows:
    print(f"{r[0]:>3} {r[1]:5} {r[2]:4}  {r[3]}  {r[4]:8}  {r[5]:8}  {str(r[6]):>12}  {str(r[7]):>3}  {str(r[8]):5}  {r[9]}")
print('--- 중앙값 (ms) ---')
for path in ('sync', 'async'):
    for cache in ('MISS', 'HIT'):
        g = [r for r in rows if r[1] == path and r[2] == cache]
        print(f"{path:5} {cache:4} | HTTP {statistics.median(r[4] for r in g):8.1f} | 최종 {statistics.median(r[5] for r in g):8.1f}")
print('저장: reports/data/timings.csv')
