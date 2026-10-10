# 🛡️ SIGAP: Smarter Protection for Sustainable JKN
**Sistem Identifikasi & Analitik Risiko Anti-Penyalahgunaan Peserta JKN**

SIGAP adalah sistem *Silent Post-Audit Screening* berbasis Hybrid AI yang beroperasi di backend untuk mendeteksi kecurangan klaim JKN (BPJS Kesehatan) tanpa mengganggu alur kerja Faskes. Sistem ini menggunakan pendekatan *Zero Disruption* dan *Human-in-the-Loop*.

Dikembangkan oleh **Tim Terserah** untuk Healthcare Domain × AI Engineering × Data Privacy.

---

##  Fitur Utama (Hybrid Intelligence)
1. **Rule-Based Engine:** Menyaring pelanggaran regulasi eksplisit (misal: Status Non-Aktif).
2. **Spatiotemporal Anomaly:** Mendeteksi *Impossible Travel* menggunakan Haversine Formula & Time Delta.
3. **Graph Network Analysis:** Membongkar sindikat terstruktur (misal: 1 No. HP terhubung ke >15 NIK).
4. **Machine Learning (Isolation Forest):** Deteksi anomali kompleks tanpa pengawasan (*unsupervised*).
5. **Explainable AI (XAI):** Setiap skor risiko disertai alasan spesifik dan *evidence trail*.

## 🛠️ Tech Stack
- **Backend & API:** FastAPI, Uvicorn
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Isolation Forest)
- **Graph Analysis:** NetworkX
- **Geospatial:** Geopy

---

##  Struktur Proyek
```text
SIGAP.1/
├── data/
│   ├── inbox/          # Input CSV klaim mentah
│   └── out/            # Output JSON/CSV hasil screening & fallback UI
├── generate_data.py    # Generator 10.000 data sintetis (UU PDP Compliant)
├── sigap_engine.py     # Otak AI: Rule, Spatiotemporal, Graph, ML
├── sigap_pipeline.py   # CLI untuk Nightly Batch Processing
├── main.py             # Server FastAPI untuk Dashboard Verifikator
└── requirements.txt    # Dependensi Python