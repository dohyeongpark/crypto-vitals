output "cluster_endpoint" {
  description = "GKE 클러스터 엔드포인트"
  value       = google_container_cluster.primary.endpoint
}

output "registry_url" {
  description = "Artifact Registry 이미지 경로"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/crypto-vitals/collector"
}
