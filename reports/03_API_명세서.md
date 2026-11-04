# 03. API 명세서

## 1. API 목록

| API ID | Method | URL | 요청 형태 | 성공 HTTP | 주요 응답 | 실패 HTTP |
|---|---|---|---|---:|---|---|
| API-01 | GET | `/health` | 없음 | 200 | `status` | 서버 미실행 시 응답 없음 |
| API-02 | POST | `/api/sync/classify/` | `multipart/form-data` / `image` | 200 | `model_id`, `top1`, `all`, `inference_ms`, `cache_hit` | 400, 413 |
| API-03 | POST | `/api/async/classify/` | `multipart/form-data` / `image` | 202 | `job_id`, `status`, `check_url` | 400, 413, 503 |
| API-04 | GET | `/api/async/jobs/{job_id}/` | 정수형 `job_id` | 200 | `job_id`, `status`, `model_id`, `top1`, `all`, `inference_ms`, `cache_hit`, `error_message` | 404 |
| WEB-01 | GET | `/vision/sync/` | 없음 | 200 | HTML (동기 화면) | — |
| WEB-02 | GET | `/vision/async/` | 없음 | 200 | HTML (비동기 화면) | — |

- 이미지 조건 (`ai/services.py`): JPEG / PNG / WEBP, 1바이트 이상 5MB 이하, 1,200만 픽셀 이하
- 503: 비동기 접수 시 Redis 브로커 등록 실패 (`router_async.py`)

## 2. API-02 동기 분류 상세

| 항목 | 내용 |
|---|---|
| URL | `/api/sync/classify/` |
| Method | POST |
| Content-Type | `multipart/form-data` |
| 요청 필드 | `image`: JPEG/PNG/WEBP, 5MB 이하 |
| 성공 코드 | 200 OK |
| 오류 코드 | 400 잘못된 이미지, 413 5MB 초과 |
| 처리 | `classify_sync()` → `classify_with_cache()` → (MISS) `classify_image_bytes()` |

**성공 응답 (실제, `bean_rust_test.0.jpg`, 캐시 MISS)**

```json
{
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "bean_rust", "score": 0.6051397323608398},
  "all": [
    {"label": "bean_rust", "score": 0.6051397323608398},
    {"label": "healthy", "score": 0.2607002258300781},
    {"label": "angular_leaf_spot", "score": 0.13415999710559845}
  ],
  "inference_ms": 17.63,
  "cache_hit": false
}
```

**캐시 HIT 응답 (실제, 같은 파일 2번째 요청)** — `"inference_ms": null, "cache_hit": true`
→ `null`은 0ms가 아니라 모델 추론을 생략했다는 뜻

**오류 응답 (실제, `.txt` 업로드)** — HTTP 400

```json
{"detail": "올바른 이미지 파일이 아닙니다."}
```

## 3. API-03 비동기 접수 상세

| 항목 | 내용 |
|---|---|
| URL | `/api/async/classify/` |
| 성공 코드 | **202 Accepted** (작업 접수만, 분석 완료 아님) |
| 처리 | `enqueue_classify()` → 이미지 검증 → `media/jobs/<uuid>.bin` 저장 → `image_jobs`(queued) 생성 → `classify_job.delay(job_id)` |
| 오류 코드 | 400 잘못된 이미지, 413 5MB 초과, 503 큐 등록 실패 |

**접수 응답 (실제)** — HTTP 202

```json
{"job_id": 1, "status": "queued", "check_url": "/api/async/jobs/1/"}
```

## 4. API-04 상태 조회 상세

| 항목 | 내용 |
|---|---|
| URL | `/api/async/jobs/{job_id}/` |
| 성공 코드 | 200 OK (`status`가 queued / running / done / failed 중 하나) |
| 처리 | `get_job()` → PostgreSQL `image_jobs` 조회 |
| 오류 코드 | 404 없는 작업 |

**완료 응답 (실제, job 1)** — HTTP 200

```json
{
  "job_id": 1,
  "status": "done",
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "bean_rust", "score": 0.6051397323608398},
  "all": [
    {"label": "bean_rust", "score": 0.6051397323608398},
    {"label": "healthy", "score": 0.2607002258300781},
    {"label": "angular_leaf_spot", "score": 0.13415999710559845}
  ],
  "inference_ms": 2555.81,
  "cache_hit": false,
  "error_message": null
}
```
(job 1은 worker의 첫 작업이라 `inference_ms`에 모델 로딩 시간 포함)

**오류 응답 (실제, job 999999)** — HTTP 404

```json
{"detail": "해당 작업이 없습니다."}
```

## 5. 실행 근거표

| 검증 ID | 테스트 입력 | 실제 상태코드 | 결과 | 증빙 |
|---|---|---:|---|---|
| API-01-T01 | GET `/health` | 200 | `{"status":"ok"}` | [health.png](evidence/health.png) |
| API-02-T01 | angular_leaf_spot / bean_rust / healthy test 각 1장 | 200 | 3장 모두 정답 클래스 | [api_sync_200.png](evidence/api_sync_200.png) |
| API-02-T02 | `bad.txt` | 400 | "올바른 이미지 파일이 아닙니다." | [bad_image.png](evidence/bad_image.png) |
| API-02-T03 | 5MB 초과 파일 | — | **미실시** (모의 테스트 T-11에서만 413 확인) | — |
| API-03-T01 | `bean_rust_test.0.jpg` | 202 | job_id 1, queued | [api_async_202.png](evidence/api_async_202.png) |
| API-04-T01 | job_id 1 | 200 | done, bean_rust 0.6051 | [api_job_done.png](evidence/api_job_done.png) |
| API-04-T02 | job_id 999999 | 404 | "해당 작업이 없습니다." | [bad_image.png](evidence/bad_image.png) |
| WEB-01 | `/vision/sync/` 업로드 | 200 | HTTP 200 / 응답까지 54ms | [ui_sync.png](evidence/ui_sync.png) |
| WEB-02 | `/vision/async/` 업로드 | 202 → 200 | 접수 24ms / 전체 1043ms / done | [ui_async.png](evidence/ui_async.png) |

- `202`는 작업 접수, 상태 조회 응답의 `done`이 작업 완료를 의미
- 응답 필드명은 `all`이며, `all_scores`는 DB 컬럼명 (산출물 4)
