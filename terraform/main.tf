# Google Cloud Provider 설정
provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. VPC 네트워크 생성
resource "google_compute_network" "vpc_network" {
  name                    = "crypto-vitals-vpc"
  auto_create_subnetworks = false
}

# 2. 서브넷 생성 (GKE용)
resource "google_compute_subnetwork" "gke_subnet" {
  name          = "gke-subnet"
  ip_cidr_range = "10.0.0.0/20"
  region        = var.region
  network       = google_compute_network.vpc_network.id
}

# 3. GKE Autopilot 클러스터 정의
resource "google_container_cluster" "primary" {
  name     = "crypto-vitals-cluster"
  location = var.region

  # Autopilot 활성화 (핵심!)
  enable_autopilot = true

  network    = google_compute_network.vpc_network.name
  subnetwork = google_compute_subnetwork.gke_subnet.name

  # 보안을 위해 공개 엔드포인트 제한 (선택 사항)
  ip_allocation_policy {
    cluster_ipv4_cidr_block  = "/14"
    services_ipv4_cidr_block = "/20"
  }

  # 취업 포트폴리오용: 리소스 가용성 보장을 위해 삭제 방지 설정
  deletion_protection = false
}

# 4. Artifact Registry 저장소 (Docker 이미지)
resource "google_artifact_registry_repository" "collector" {
  location      = var.region
  repository_id = "crypto-vitals"
  format        = "DOCKER"
}

# 5. BigQuery 데이터셋
resource "google_bigquery_dataset" "crypto_vitals" {
  dataset_id = "crypto_vitals"
  location   = var.region
}

# 6. BigQuery 테이블 (OHLCV + 변동성)
resource "google_bigquery_table" "ohlcv" {
  dataset_id          = google_bigquery_dataset.crypto_vitals.dataset_id
  table_id            = "ohlcv"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  schema = jsonencode([
    { name = "symbol",        type = "STRING",    mode = "REQUIRED" },
    { name = "timestamp",     type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "open",          type = "FLOAT64",   mode = "REQUIRED" },
    { name = "high",          type = "FLOAT64",   mode = "REQUIRED" },
    { name = "low",           type = "FLOAT64",   mode = "REQUIRED" },
    { name = "close",         type = "FLOAT64",   mode = "REQUIRED" },
    { name = "volume",        type = "FLOAT64",   mode = "REQUIRED" },
    { name = "volatility_5m", type = "FLOAT64",   mode = "NULLABLE" },
    { name = "collected_at",  type = "TIMESTAMP", mode = "REQUIRED" },
  ])
}

# 7. GCP Service Account (collector Pod용)
resource "google_service_account" "collector" {
  account_id   = "collector-sa"
  display_name = "Collector Service Account"
}

# 8. BigQuery 쓰기 권한 부여
resource "google_bigquery_dataset_iam_member" "collector_bq_writer" {
  dataset_id = google_bigquery_dataset.crypto_vitals.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.collector.email}"
}

# 9. Workload Identity 바인딩 (K8s SA → GCP SA)
resource "google_service_account_iam_member" "workload_identity" {
  service_account_id = google_service_account.collector.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[default/collector]"
}