provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Artifact Registry (Docker 이미지)
resource "google_artifact_registry_repository" "collector" {
  location      = var.region
  repository_id = "crypto-vitals"
  format        = "DOCKER"
}

# 2. BigQuery 데이터셋
resource "google_bigquery_dataset" "crypto_vitals" {
  dataset_id = "crypto_vitals"
  location   = var.region
}

# 3. BigQuery 테이블 (OHLCV + 변동성, DAY 파티셔닝)
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

# 4. GCP Service Account (collector VM용)
resource "google_service_account" "collector" {
  account_id   = "collector-sa"
  display_name = "Collector Service Account"
}

resource "google_bigquery_dataset_iam_member" "collector_bq_writer" {
  dataset_id = google_bigquery_dataset.crypto_vitals.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.collector.email}"
}

# 5. Secret Manager API 활성화
resource "google_project_service" "secretmanager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

# 6. Binance API 키 Secret 등록
resource "google_secret_manager_secret" "binance_api_key" {
  secret_id  = "binance-api-key"
  depends_on = [google_project_service.secretmanager]
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "binance_api_key" {
  secret      = google_secret_manager_secret.binance_api_key.id
  secret_data = var.binance_api_key
}

resource "google_secret_manager_secret" "binance_api_secret" {
  secret_id  = "binance-api-secret"
  depends_on = [google_project_service.secretmanager]
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "binance_api_secret" {
  secret      = google_secret_manager_secret.binance_api_secret.id
  secret_data = var.binance_api_secret
}

resource "google_project_iam_member" "collector_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.collector.email}"
}

# 7. GCS 버킷 (collector 코드 저장)
resource "google_storage_bucket" "code" {
  name          = "${var.project_id}-collector-code"
  location      = "US"
  force_destroy = true
}

resource "google_storage_bucket_object" "main_py" {
  name   = "main.py"
  bucket = google_storage_bucket.code.name
  source = "../main.py"
}

resource "google_storage_bucket_object" "requirements_txt" {
  name   = "requirements.txt"
  bucket = google_storage_bucket.code.name
  source = "../requirements.txt"
}

resource "google_storage_bucket_iam_member" "collector_storage_reader" {
  bucket = google_storage_bucket.code.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.collector.email}"
}

# 8. GCE e2-micro (무료 티어, us-central1)
resource "google_compute_instance" "collector" {
  name         = "collector-vm"
  machine_type = "e2-micro"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 10
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  service_account {
    email  = google_service_account.collector.email
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = templatefile("${path.module}/startup.sh.tpl", {
    bucket_name = google_storage_bucket.code.name
    project_id  = var.project_id
  })

  depends_on = [
    google_storage_bucket_object.main_py,
    google_storage_bucket_object.requirements_txt,
    google_secret_manager_secret_version.binance_api_key,
    google_secret_manager_secret_version.binance_api_secret,
    google_project_iam_member.collector_secret_accessor,
    google_storage_bucket_iam_member.collector_storage_reader,
  ]
}
