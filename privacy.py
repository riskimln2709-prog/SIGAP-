"""Anonimisasi NIK: nik_hash = SHA256(SALT + NIK), diambil 16 karakter (sesuai Laporan Arsitektur v3.0)."""
import hashlib, os

# Di sistem nyata SALT disimpan sebagai rahasia (variabel lingkungan), bukan di kode.
SALT = os.environ.get("SIGAP_SALT", "sigap-demo-salt-ganti-di-produksi")


def hash_nik(nik: str) -> str:
    digits = "".join(ch for ch in str(nik) if ch.isdigit())
    return hashlib.sha256((SALT + digits).encode()).hexdigest()[:16]