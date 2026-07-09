# crypto-vitals — BTC/ETH Stat Arb ML Pipeline (Phase 1–5)

BTC/ETH 공적분 기반 평균회귀 신호를 위한 GCP 데이터/ML 파이프라인.  
TimescaleDB 적재 → 칼만 필터 동적 β/OU 파라미터 → triple-barrier 라벨링 →
LightGBM 메타레이블 모델 → 표본외(OOS) 검증 및 백테스트까지 end-to-end로 구성되어 있습니다.

---

## ⚠️ 비용 주의사항

> **무료 등급 조건은 GCP 콘솔에서 직접 확인하세요.**  
> [https://cloud.google.com/free](https://cloud.google.com/free) — 조건은 변경될 수 있습니다.

현재 설정 기준 (2026-06 시점):
- `e2-micro` 무료 등급: **us-west1, us-central1, us-east1** 리전에서만 적용
- `pd-standard` 디스크: 월 30 GB까지 무료 (boot + data 합산)
- 네트워크 egress: 미주 리전 간 월 1 GB 무료
- **한국(asia-northeast3) e2-micro는 무료 아님** — 기존 VM이 있다면 `destroy` 필요

비용 안전장치:
1. `google_billing_budget` — 월 예산 초과 시 청구 계정 관리자에게 이메일 알림
2. 프로젝트 종료 시 반드시 `terraform destroy` 실행

---

## 아키텍처

```
[Binance Vision CDN]
        │
  ┌─────▼──────────────────────────────────┐
  │  GCP e2-micro VM (us-central1)          │
  │                                         │
  │  docker-compose                         │
  │   └── TimescaleDB (PostgreSQL 16)       │
  │         /data/timescaledb (분리 디스크)  │
  └─────────────────────────────────────────┘
```

> **OI 수집기 비활성**: us-central1(미국 IP)에서 Binance 선물 API가 HTTP 451로 영구 차단됨.
> `oi_scheduler` 서비스는 `docker-compose.yml`에 주석 처리되어 있으며, 향후 ablation/프록시 시 재활성 가능.
> 레짐 피처(basis, taker_ratio, funding_spread)로 OI 정보를 대체.

**레이어 분리**: Terraform = GCP 인프라만. 앱 = VM 안에서 docker-compose.

파이프라인(`src/pipeline.py`)이 적재부터 라벨링까지 8단계로 오케스트레이션되며,
그 위에 독립 실행 가능한 ML 학습/검증/백테스트 스크립트(`src/ml/`, `src/backtest/`)가 얹혀지는 구조입니다.

---

## 1. GCP / Terraform 셋업

### 사전 요구사항

```bash
# gcloud CLI 설치 및 로그인
gcloud auth application-default login

# terraform 설치 (https://developer.hashicorp.com/terraform/install)
terraform -version  # >= 1.5
```

### tfvars 작성

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars
# 편집기로 project_id, billing_account_id 등 입력
```

> `infra/terraform.tfvars`는 `.gitignore`에 포함돼 있어 커밋되지 않습니다.

billing_account_id 확인:
```bash
gcloud billing accounts list
```

### GCS remote state 버킷 생성 (권장)

```bash
gsutil mb -p <PROJECT_ID> -l us-central1 gs://<PROJECT_ID>-tfstate
```

`infra/backend.tf`의 주석을 해제한 뒤:
```bash
cd infra && terraform init -migrate-state
```

### Plan & Apply

```bash
cd infra
terraform init
terraform plan          # 확인 후
terraform apply         # 적용 (VM 생성 = 비용 발생 시작)
```

### VM SSH 접속

```bash
# outputs 에서 ssh_command 확인
terraform output ssh_command
gcloud compute ssh statarb-vm --zone=us-central1-a --project=<PROJECT_ID>
```

---

## 2. VM 초기 설정 (SSH 접속 후)

Startup script가 Docker + docker-compose를 설치하고 `/data` 디스크를 마운트합니다.
완료 확인 (수 분 소요):

```bash
systemctl is-active docker
ls /data/timescaledb
```

프로젝트 코드 복사:

```bash
git clone <repo> ~/crypto-vitals
cd ~/crypto-vitals
cp .env.example .env
# .env 편집: DB_PASSWORD 등 설정
```

TimescaleDB 기동:

```bash
docker compose up -d timescaledb
docker compose logs -f timescaledb   # healthy 확인
```

---

## 3. Track A — 데이터 적재 · 피처 · 라벨링 파이프라인

> **처음에는 1개월로 end-to-end 검증 후 3년 전체로 확장하세요.**

### Step 1: 1개월 검증

```bash
python -m src.pipeline --months 1
```

### Step 2: 3년 전체 적재

```bash
python -m src.pipeline --start 2023-01
```

### Step 3: 증분 실행 (일일 갱신 등, 이미 적재된 구간은 건너뜀)

```bash
python -m src.pipeline --start 2023-01 --incremental
```

### 단계별 개별 실행

```bash
# A-1: klines 다운로드 (data.binance.vision)
python -m src.collectors.klines_bulk --start 2023-01 --end 2023-02

# A-2: 펀딩비 수집 및 forward-fill
python -m src.collectors.funding --start 2023-01-01

# A-3: 무결성 검증
python -m src.features.integrity

# A-4: 실현변동성 계산
python -m src.features.volatility

# A-5: 페어 스프레드 / z-score
python -m src.features.pair_spread

# A-6: 레짐 피처 (basis / taker_ratio / funding_spread)
python -m src.features.regime

# A-7: 칼만 필터 동적 β / OU 파라미터(θ, 반감기) / 공적분 검정
python -m src.features.kalman_ou

# A-8: triple-barrier ML 라벨 생성 (ou_zscore 기반 진입 신호)
python -m src.labels.triple_barrier
```

`--skip-*` 플래그(`--skip-klines`, `--skip-kalman`, `--skip-labels` 등)로 이미 완료한 단계를 건너뛸 수 있습니다.

---

## 4. Track B — OI 수집 (현재 비활성)

> us-central1 IP에서 Binance 선물 API(`fapi.binance.com`)가 HTTP 451로 영구 차단됨.
> `oi_scheduler` 서비스는 비활성 상태. `open_interest` 테이블 스키마는 보존.
>
> 대체 레짐 피처: `basis`, `taker_ratio`, `funding_spread` — Track A-6으로 제공.

---

## 5. ML 메타레이블 학습 (Phase 3)

triple-barrier 라벨(`ml_labels`)과 `pair_features`를 조인해 LightGBM 이진 분류기를 학습합니다.
Purged walk-forward CV로 룩어헤드 없이 검증합니다.

```bash
python -m src.ml.train
python -m src.ml.train --label-version v1.1-tb --feature-version v0.3-kalman
```

학습된 모델은 `models/`에 저장됩니다.

### SHAP 피처 중요도

```bash
python -m src.ml.explain
```

### 하이퍼파라미터 그리드서치 (Phase 4)

TB 진입/청산 z-score × LightGBM 파라미터 조합(8×9=72개)을 purged CV로 탐색해
평균 순수익 기준 최적 조합을 찾습니다.

```bash
python -m src.ml.hparam
```

현재 기본값(`src/config.py`)은 그리드서치 최적값(`entry_z=2.5`, `stop_z=3.5`)이 반영된 `v1.1-tb`.

---

## 6. 백테스트 (Phase 4)

메타레이블 예측(`meta_prob`)을 이용한 P&L 시뮬레이션. 수수료·슬리피지 반영.

```bash
python -m src.backtest
python -m src.backtest --fee 0.004 --slippage 0.002 --label-version v1.1-tb
```

---

## 7. 표본외(OOS) 검증 (Phase 5)

`holdout_start` 이전 데이터로만 학습하고, 이후 구간에서 평가합니다.
`exit_timestamp`가 holdout 경계를 넘는 학습 샘플은 purge하여 경계를 넘는 룩어헤드를 차단합니다.

```bash
python -m src.ml.oos_eval
python -m src.ml.oos_eval --label-version v1.1-tb --holdout-start 2025-01-01
```

---

## 8. 테스트 실행

```bash
pip install pytest
pytest tests/ -v
```

테스트 파일 구성 (총 28개, 전부 통과):
- `tests/test_lookahead.py` — 룩어헤드 방지 검증: `timestamp = close_time` 확인, 롤링 β/z-score/실현변동성/칼만 필터/레짐 피처가 미래 데이터를 참조하지 않음, `funding_rate` forward-fill 방향성
- `tests/test_backtest.py` — 백테스트 P&L/수수료·슬리피지 공식 검증
- `tests/test_oos.py` — OOS 평가의 시간 경계 purge 정확성 검증
- `tests/test_incremental.py` — `--incremental` 파이프라인 실행의 멱등성 검증

---

## 9. Binance API 키

**이 프로젝트는 API 키가 필요 없습니다.**  
사용하는 모든 엔드포인트는 공개(public) 데이터입니다:
- `data.binance.vision` — 과거 klines ZIP, 월별 펀딩비 ZIP (인증 불필요, 지역 제한 없음)

> `/fapi/*`, `/futures/data/*` 엔드포인트는 us-central1 IP에서 HTTP 451 차단.
> klines·펀딩비 모두 CDN에서 수집하므로 API 키·프록시 모두 불필요.

---

## 10. 프로젝트 종료 시

과금이 계속되지 않도록 반드시 리소스를 삭제하세요:

```bash
# 앱 중단
docker compose down

# GCP 리소스 삭제 (VM, 디스크, 방화벽, 예산 알림 모두 삭제)
cd infra
terraform destroy
```

> **영속 디스크(`statarb-data-disk`)의 데이터도 삭제됩니다.**  
> 데이터를 보존하려면 destroy 전에 스냅샷을 찍으세요:
> ```bash
> gcloud compute disks snapshot statarb-data-disk --zone=us-central1-a
> ```

---

## 디렉토리 구조

```
.
├── infra/                  # Terraform (GCP 인프라, GCS remote state)
├── db/
│   ├── schema.sql          # TimescaleDB 테이블 정의
│   ├── migration_v02.sql   # v0.2-regime 컬럼 추가
│   ├── migration_v03.sql   # v0.3-kalman: 칼만 β/OU 컬럼 11개 추가
│   └── migration_v04.sql   # ml_labels 스키마 (triple-barrier 라벨)
├── docker-compose.yml      # TimescaleDB (oi_scheduler는 비활성 주석)
├── Dockerfile              # Python app 컨테이너
├── .env.example
├── requirements.txt
├── models/                 # 학습된 LightGBM 모델 / SHAP / hparam 결과 저장
├── src/
│   ├── config.py
│   ├── db.py               # 연결 풀, UPSERT/UPDATE 헬퍼, SQLAlchemy engine
│   ├── collectors/
│   │   ├── klines_bulk.py  # A-1: 과거 OHLCV (CDN)
│   │   ├── funding.py      # A-2: 펀딩비 (CDN)
│   │   └── oi_scheduler.py # Track B: OI (현재 비활성 — HTTP 451)
│   ├── features/
│   │   ├── integrity.py    # A-3: 결측봉 탐지
│   │   ├── volatility.py   # A-4: 실현변동성
│   │   ├── pair_spread.py  # A-5: 롤링 OLS β, MAD z-score
│   │   ├── regime.py       # A-6: basis / taker_ratio / funding_spread
│   │   └── kalman_ou.py    # A-7: 칼만 필터 동적 β / OU 파라미터 / 공적분 검정
│   ├── labels/
│   │   └── triple_barrier.py # A-8: triple-barrier ML 라벨링
│   ├── ml/
│   │   ├── features.py     # 학습용 피처 매트릭스 구성
│   │   ├── cv.py           # purged walk-forward CV
│   │   ├── train.py        # LightGBM 메타레이블 학습
│   │   ├── predict.py      # 예측 → ml_labels.meta_prob 갱신
│   │   ├── explain.py      # SHAP 피처 중요도
│   │   ├── hparam.py       # TB + LightGBM 하이퍼파라미터 그리드서치
│   │   └── oos_eval.py     # 표본외(OOS) 검증
│   ├── backtest/
│   │   ├── engine.py       # P&L 시뮬레이션
│   │   ├── metrics.py      # 성과 지표
│   │   └── __main__.py     # 백테스트 CLI
│   └── pipeline.py         # Track A 오케스트레이션 (8단계, --incremental 지원)
└── tests/                  # 28개 테스트 (전부 통과)
    ├── test_lookahead.py   # 룩어헤드 방지 검증
    ├── test_backtest.py    # 백테스트 공식 검증
    ├── test_oos.py         # OOS 경계 purge 검증
    └── test_incremental.py # 증분 파이프라인 멱등성 검증
```

---

## Phase 로드맵

| Phase | 내용 | 상태 |
|-------|------|------|
| 1 | GCP 인프라 + 3년 OHLCV + 레짐 피처(basis/taker/funding) | ✅ 완료 |
| 2 | 칼만 필터 동적 β + 공적분 검정(EG/Johansen) + OU 파라미터(θ, 반감기) + 진입·청산 임계값 | ✅ 완료 |
| 3 | Triple-barrier 라벨링 + ML 메타레이블(LightGBM) | ✅ 완료 |
| 4 | 하이퍼파라미터 그리드서치 + SHAP 피처 중요도 + 거래비용 반영 백테스트 | ✅ 완료 |
| 5 | 피처 프루닝(19→13) + 엄격한 시간 경계 표본외(OOS) 검증 + 증분 파이프라인(`--incremental`) | ✅ 완료 |
