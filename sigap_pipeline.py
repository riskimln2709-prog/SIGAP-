import argparse, os, time
import pandas as pd
from sigap_engine import SIGAPEngine


def jalankan(inbox, out):
    files = sorted(f for f in os.listdir(inbox) if f.endswith(".csv"))
    if not files:
        print("[SIGAP] Tidak ada file CSV di inbox. Selesai."); return
    # semua CSV digabung (tidak lagi saling menimpa); duplikat ID dibuang
    df = pd.concat([pd.read_csv(os.path.join(inbox, f)) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset="ID_Klaim", keep="last")
    print(f"[SIGAP] {len(files)} file, {len(df)} klaim dibaca.")
    engine = SIGAPEngine()
    hasil = engine.run_pipeline(df)
    alert = hasil[hasil["risk_category"] != "Aman (Hijau)"].sort_values("risk_score", ascending=False)
    os.makedirs(out, exist_ok=True)
    alert.to_json(os.path.join(out, "dashboard_feed.json"), orient="records", date_format="iso")
    hasil.to_csv(os.path.join(out, "full_audit_log.csv"), index=False)
    n = hasil["risk_category"].value_counts()
    print(f"[SIGAP] Selesai. Merah: {n.get('Risiko Tinggi (Merah)', 0)}, Kuning: {n.get('Review Ringan (Kuning)', 0)}, "
          f"Aman: {n.get('Aman (Hijau)', 0)}. {len(alert)} klaim masuk antrean verifikator.")


def main():
    p = argparse.ArgumentParser(description="SIGAP Nightly Batch Processor")
    p.add_argument("mode", choices=["nightly"])
    p.add_argument("--inbox", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--once", action="store_true", help="Jalankan sekali lalu berhenti")
    p.add_argument("--interval-menit", type=int, default=1440, help="Jeda antar-proses jika tanpa --once (bawaan 24 jam)")
    a = p.parse_args()
    print("[SIGAP] Memulai Nightly Batch Processing...")
    while True:
        jalankan(a.inbox, a.out)
        if a.once:
            break
        print(f"[SIGAP] Menunggu {a.interval_menit} menit untuk proses berikutnya (Ctrl+C untuk berhenti)...")
        time.sleep(a.interval_menit * 60)


if __name__ == "__main__":
    main()