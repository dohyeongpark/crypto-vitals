output "vm_external_ip" {
  description = "VM 외부 IP (동적 IP는 재시작마다 바뀜)"
  value       = google_compute_instance.statarb.network_interface[0].access_config[0].nat_ip
}

output "ssh_command" {
  description = "VM SSH 접속 명령어"
  value       = "gcloud compute ssh statarb-vm --zone=${var.region}-${var.zone_suffix} --project=${var.project_id}"
}

output "data_disk_name" {
  description = "영속 데이터 디스크 이름 (timescaledb 데이터 저장)"
  value       = google_compute_disk.data.name
}

output "static_ip" {
  description = "고정 외부 IP (use_static_ip=true 일 때만)"
  value       = var.use_static_ip ? google_compute_address.vm[0].address : "N/A (dynamic IP)"
}
