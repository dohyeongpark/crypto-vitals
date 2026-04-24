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