terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.5"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── Static IP (optional) ─────────────────────────────────────────────────────
resource "google_compute_address" "vm" {
  count  = var.use_static_ip ? 1 : 0
  name   = "statarb-vm-ip"
  region = var.region
}

# ── Persistent data disk (survives VM replacement) ────────────────────────────
# TimescaleDB data lives here, not on the boot disk.
resource "google_compute_disk" "data" {
  name = "statarb-data-disk"
  type = "pd-standard"
  zone = "${var.region}-${var.zone_suffix}"
  size = var.data_disk_gb
}

# ── VM (e2-micro, free-tier eligible in US regions) ──────────────────────────
resource "google_compute_instance" "statarb" {
  name         = "statarb-vm"
  machine_type = var.machine_type
  zone         = "${var.region}-${var.zone_suffix}"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = var.boot_disk_gb
    }
  }

  # Attach persistent data disk as a secondary disk.
  attached_disk {
    source      = google_compute_disk.data.self_link
    device_name = "statarb-data"
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = var.use_static_ip ? google_compute_address.vm[0].address : null
    }
  }

  # Startup script only installs Docker + docker-compose.
  # App code is deployed manually via SSH during active development.
  metadata_startup_script = file("${path.module}/startup.sh.tpl")

  tags = ["statarb-vm"]

  # Ensure disk is ready before VM starts.
  depends_on = [google_compute_disk.data]
}

# ── Firewall: SSH only, scoped to your IP ────────────────────────────────────
resource "google_compute_firewall" "ssh" {
  name    = "statarb-allow-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  # Restrict to a specific CIDR. Set allowed_ssh_cidr = "0.0.0.0/0" to allow
  # all (less secure). DB port (5432) is intentionally NOT exposed externally.
  source_ranges = [var.allowed_ssh_cidr]
  target_tags   = ["statarb-vm"]
}

# ── Billing budget alert ──────────────────────────────────────────────────────
# Sends email to billing account admins when thresholds are crossed.
# This is the primary cost safety mechanism — do NOT delete.
resource "google_billing_budget" "statarb" {
  billing_account = var.billing_account_id
  display_name    = "statarb-monthly-budget"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_amount_usd)
    }
  }

  # Alert at 50%, 80%, 100% of monthly budget.
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.8
  }
  threshold_rules {
    threshold_percent = 1.0
    spend_basis       = "FORECASTED_SPEND"
  }

  # GCP sends email to billing account admins automatically when thresholds
  # are crossed. No pubsub/monitoring channel needed for basic email alerts.
}
