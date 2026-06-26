variable "project_id" {
  description = "GCP 프로젝트 ID (예: parkdh0121)"
  type        = string
}

variable "project_number" {
  description = "GCP 프로젝트 번호 (숫자). billing budget filter에 필요. gcloud projects describe <id> --format=value(projectNumber)"
  type        = string
}

variable "billing_account_id" {
  description = "GCP 청구 계정 ID (형식: XXXXXX-XXXXXX-XXXXXX). GCP 콘솔 > 청구 > 계정 관리에서 확인."
  type        = string
}

# ── Region / Zone ─────────────────────────────────────────────────────────────
# IMPORTANT: e2-micro 무료 등급은 us-west1, us-central1, us-east1 리전에서만 적용.
# 한국(asia-northeast3) 등 다른 리전으로 변경하면 과금 발생.
variable "region" {
  description = "GCP 리전 (e2-micro 무료 등급: us-west1 | us-central1 | us-east1)"
  type        = string
  default     = "us-central1"
}

variable "zone_suffix" {
  description = "Zone suffix appended to region (e.g. 'a' → us-central1-a)"
  type        = string
  default     = "a"
}

# ── VM ────────────────────────────────────────────────────────────────────────
variable "machine_type" {
  description = "GCE machine type. e2-micro = 무료 등급 대상."
  type        = string
  default     = "e2-micro"
}

variable "boot_disk_gb" {
  description = "Boot disk size (GB). 무료 등급 총 30 GB 한도 안에서 data_disk_gb와 합산."
  type        = number
  default     = 15
}

variable "data_disk_gb" {
  description = "Persistent data disk size (GB) for TimescaleDB. VM 교체 시에도 유지됨."
  type        = number
  default     = 15
}

# ── Network ───────────────────────────────────────────────────────────────────
variable "use_static_ip" {
  description = "true = 고정 외부 IP 할당 (월 ~$1.5). false = 동적 IP (재시작 시 변경됨)."
  type        = bool
  default     = false
}

variable "allowed_ssh_cidr" {
  description = "SSH 허용 IP CIDR. 본인 IP로 제한 권장 (예: '1.2.3.4/32'). '0.0.0.0/0' = 전체 허용."
  type        = string
  default     = "0.0.0.0/0"
}

# ── Budget ────────────────────────────────────────────────────────────────────
variable "budget_amount" {
  description = "월간 예산 상한 (budget_currency_code 단위). 초과 시 청구 계정 관리자에게 이메일 알림."
  type        = number
  default     = 15000
}

variable "budget_currency_code" {
  description = "청구 계정의 통화 코드. gcloud billing budgets create 시 표시되는 currencyCode와 일치해야 함."
  type        = string
  default     = "KRW"
}
