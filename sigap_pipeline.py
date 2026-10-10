import argparse
import os
import pandas as pd
from sigap_engine import SIGAPEngine
import json

def main():
    parser = argparse.ArgumentParser(description="SIGAP Nightly Batch Processor")
    parser.add_argument('mode', choices=['nightly'], help="Mode eksekusi")
    parser.add_argument('--inbox', required=True, help="Path folder input CSV")
    parser.add_argument('--out', required=True, help="Path folder output JSON/CSV")
    parser.add_argument('--once', action='store_true', help="Jalankan sekali lalu exit")
    
    args = parser.parse_args()
    
    if args.mode == 'nightly':
        print("[SIGAP] Memulai Nightly Batch Processing...")
        engine = SIGAPEngine()
        
        # Baca semua CSV di folder inbox
        all_files = [f for f in os.listdir(args.inbox) if f.endswith('.csv')]
        
        if not all_files:
            print("[SIGAP] Tidak ada file baru di inbox. Selesai.")
            return

        for file in all_files:
            print(f"[SIGAP] Memproses {file}...")
            df = pd.read_csv(os.path.join(args.inbox, file))
            
            # Jalankan Engine AI
            df_result = engine.run_pipeline(df)
            
            # Simpan hasil (Filter hanya yang Merah & Kuning untuk Dashboard)
            df_alert = df_result[df_result['risk_category'] != 'Aman (Hijau)']
            
            # Export ke JSON untuk API
            out_path_json = os.path.join(args.out, 'dashboard_feed.json')
            df_alert.to_json(out_path_json, orient='records', date_format='iso')
            
            # Export full CSV untuk arsip
            out_path_csv = os.path.join(args.out, 'full_audit_log.csv')
            df_result.to_csv(out_path_csv, index=False)
            
            print(f"[SIGAP] Berhasil memproses {len(df)} klaim. {len(df_alert)} klaim masuk antrean verifikator.")
            
            # Pindahkan file yang sudah diproses (opsional, bisa dihapus atau di-archive)
            # os.rename(os.path.join(args.inbox, file), os.path.join(args.inbox, 'processed_' + file))

        print("[SIGAP] Nightly Batch Selesai.")

if __name__ == "__main__":
    main()