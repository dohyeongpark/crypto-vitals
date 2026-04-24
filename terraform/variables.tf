variable "project_id" {
  description = "GCP 프로젝트 ID"
  type        = string
}

variable "region" {
  description = "GCP 리전"
  type        = string
  default     = "asia-northeast3"
}

variable "cluster_name" {
  description = "GKE 클러스터 이름"
  type        = string
  default     = "crypto-vitals-cluster"
}

variable "binance_api_key" {
  description = "Binance API 키"
  type        = string
  sensitive   = true
}

variable "binance_api_secret" {
  description = "Binance API 시크릿"
  type        = string
  sensitive   = true
}
