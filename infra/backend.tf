# ── Remote state backend (GCS) ───────────────────────────────────────────────
#
# 로컬 state는 노트북에 묶여 위험합니다. GCS 백엔드를 권장합니다.
#
# 사용 방법:
#   1. 아래 주석을 해제합니다.
#   2. GCS 버킷을 먼저 수동으로 생성합니다:
#      gsutil mb -p <PROJECT_ID> -l us-central1 gs://<PROJECT_ID>-tfstate
#   3. terraform init -migrate-state  (기존 로컬 state → GCS 이전)
#
# terraform {
#   backend "gcs" {
#     bucket = "<PROJECT_ID>-tfstate"
#     prefix = "statarb/terraform.tfstate"
#   }
# }
