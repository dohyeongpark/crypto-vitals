# crypto-vitals — BTC/ETH Stat Arb Data Pipeline (Phase 1)

BTC/ETH 공적분 기반 평균회귀 신호를 위한 GCP 데이터 인프라.  
TimescaleDB + 과거 3년 OHLCV + 레짐 피처(basis / taker ratio / funding spread) 적재 파이프라인.

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

## 3. Track A — 과거 데이터 적재

> **처음에는 1개월로 end-to-end 검증 후 3년 전체로 확장하세요.**

### Step 1: 1개월 검증

```bash
python -m src.pipeline --months 1
```

### Step 2: 3년 전체 적재

```bash
python -m src.pipeline --start 2023-01
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
```

---

## 4. Track B — OI 수집 (현재 비활성)

> us-central1 IP에서 Binance 선물 API(`fapi.binance.com`)가 HTTP 451로 영구 차단됨.
> `oi_scheduler` 서비스는 비활성 상태. `open_interest` 테이블 스키마는 보존.
>
> 대체 레짐 피처: `basis`, `taker_ratio`, `funding_spread` — Track A-6으로 제공.

---

## 5. 테스트 실행

```bash
pip install pytest
pytest tests/test_lookahead.py -v
```

룩어헤드 검증 내용 (7개 테스트):
- `timestamp = close_time` (open_time 아님) 확인
- 롤링 β, z-score, 실현변동성이 미래 데이터를 참조하지 않음
- `basis` 계산이 동시각 spot/perp 값만 사용
- `funding_rate` forward-fill이 과거→미래 방향으로만 전파

---

## 6. Binance API 키

**이 프로젝트는 API 키가 필요 없습니다.**  
사용하는 모든 엔드포인트는 공개(public) 데이터입니다:
- `data.binance.vision` — 과거 klines ZIP, 월별 펀딩비 ZIP (인증 불필요, 지역 제한 없음)

> `/fapi/*`, `/futures/data/*` 엔드포인트는 us-central1 IP에서 HTTP 451 차단.
> klines·펀딩비 모두 CDN에서 수집하므로 API 키·프록시 모두 불필요.

---

## 7. 프로젝트 종료 시

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
├── infra/                  # Terraform (GCP 인프라)
├── db/
│   ├── schema.sql          # TimescaleDB 테이블 정의
│   └── migration_v02.sql   # v0.2-regime 컬럼 추가 마이그레이션
├── docker-compose.yml      # TimescaleDB (oi_scheduler는 비활성 주석)
├── Dockerfile              # Python app 컨테이너
├── .env.example
├── requirements.txt
├── src/
│   ├── config.py
│   ├── db.py               # 연결 풀, UPSERT/UPDATE 헬퍼, SQLAlchemy engine
│   ├── collectors/
│   │   ├── klines_bulk.py  # Track A-1: 과거 OHLCV (CDN)
│   │   ├── funding.py      # Track A-2: 펀딩비 (CDN)
│   │   └── oi_scheduler.py # Track B: OI (현재 비활성 — HTTP 451)
│   ├── features/
│   │   ├── integrity.py    # Track A-3: 결측봉 탐지
│   │   ├── volatility.py   # Track A-4: 실현변동성
│   │   ├── pair_spread.py  # Track A-5: 롤링 OLS β, MAD z-score
│   │   └── regime.py       # Track A-6: basis / taker_ratio / funding_spread
│   └── pipeline.py         # Track A 오케스트레이션 (6단계)
└── tests/
    └── test_lookahead.py   # 룩어헤드 없음 검증 (7개 테스트)
```

---

## Phase 로드맵

| Phase | 내용 | 상태 |
|-------|------|------|
| 1 | GCP 인프라 + 3년 OHLCV + 레짐 피처(basis/taker/funding) | ✅ 완료 |
| 2 | 칼만 필터 동적 β + 공적분 검정(EG/Johansen) + OU 파라미터(θ, 반감기) + 진입·청산 임계값 | 예정 |
| 3 | Triple-barrier 라벨링 + ML 메타레이블(LightGBM) | 예정 |
| 4 | Purged CV / Deflated Sharpe 검증 + 거래비용 모델 백테스트 | 예정 |
