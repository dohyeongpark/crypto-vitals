output "registry_url" {
  description = "Artifact Registry 이미지 경로"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/crypto-vitals/collector"
}

output "collector_sa_email" {
  description = "Collector GCP Service Account 이메일"
  value       = google_service_account.collector.email
}

output "collector_vm_ip" {
  description = "Collector VM 외부 IP"
  value       = google_compute_instance.collector.network_interface[0].access_config[0].nat_ip
}

output "collector_vm_ssh" {
  description = "VM SSH 접속 명령어"
  value       = "gcloud compute ssh collector-vm --zone=us-central1-a --project=${var.project_id}"
}
