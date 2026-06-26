# ── Remote state backend (GCS) ───────────────────────────────────────────────
# 버킷: gs://parkdh0121-tfstate (versioning + uniform access 활성화)
# 로컬 state에서 이전: terraform init -migrate-state

terraform {
  backend "gcs" {
    bucket = "parkdh0121-tfstate"
    prefix = "statarb/terraform.tfstate"
  }
}
