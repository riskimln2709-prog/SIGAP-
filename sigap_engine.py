import pandas as pd
import numpy as np
import networkx as nx
from geopy.distance import geodesic
from sklearn.ensemble import IsolationForest
from datetime import datetime


class SIGAPEngine:
    def __init__(self):
        # Inisialisasi Model Isolation Forest untuk Spatiotemporal Anomaly
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.is_fitted = False

    def _calculate_haversine(self, row):
        """Hitung jarak fisik antara klaim saat ini dan klaim sebelumnya"""
        try:
            # ✅ PERBAIKAN: coord1 = posisi SAAT INI, coord2 = posisi SEBELUMNYA
            coord1 = (row['Lat_Faskes'], row['Lng_Faskes'])
            coord2 = (row['Lat_Faskes_1'], row['Lng_Faskes_1'])
            
            # Jika salah satu NaN (klaim pertama), return 0
            if pd.isna(coord1[0]) or pd.isna(coord2[0]):
                return 0
                
            return geodesic(coord1, coord2).km
        except Exception:
            return 0

    def _calculate_speed(self, row):
        """Hitung kecepatan perpindahan (km/jam)"""
        if row['time_delta_hours'] > 0:
            return row['distance_km'] / row['time_delta_hours']
        return 9999  # Jika waktu 0 tapi jarak > 0, kecepatan tak terhingga (Fraud!)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lapis 1 & 2: Rule Engine & Spatiotemporal Features"""
        df = df.copy()

        # 1. Rule-Based Engine (Filter Regulasi)
        df['rule_penalty'] = 0
        df.loc[df['Status_Aktif'] == 'Non-Aktif', 'rule_penalty'] = 50

        # 2. Spatiotemporal Features
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # ✅ PERBAIKAN: Sort dulu berdasarkan NIK dan Timestamp
        df = df.sort_values(by=['NIK_Hash', 'Timestamp'])

        # ✅ PERBAIKAN KRITIS: Shift PER NIK, bukan global!
        # Agar tidak mencampur data pasien berbeda
        df['Lat_Faskes_1'] = df.groupby('NIK_Hash')['Lat_Faskes'].shift(1)
        df['Lng_Faskes_1'] = df.groupby('NIK_Hash')['Lng_Faskes'].shift(1)
        df['Timestamp_1'] = df.groupby('NIK_Hash')['Timestamp'].shift(1)

        # Hitung Jarak dan Waktu
        df['distance_km'] = df.apply(self._calculate_haversine, axis=1)
        df['time_delta_hours'] = (df['Timestamp'] - df['Timestamp_1']).dt.total_seconds() / 3600
        df['speed_kmh'] = df.apply(self._calculate_speed, axis=1)

        # Handle NaN dari shift pertama (klaim pertama per NIK tidak punya "klaim sebelumnya")
        df[['distance_km', 'time_delta_hours', 'speed_kmh']] = df[['distance_km', 'time_delta_hours', 'speed_kmh']].fillna(0)

        return df

    def build_graph_syndicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lapis 3: Graph Network Analysis untuk deteksi sindikat"""
        # Group by No_HP untuk melihat berapa banyak NIK unik yang terhubung
        hp_nik_mapping = df.groupby('No_HP_Daftar')['NIK_Hash'].nunique().reset_index()
        hp_nik_mapping.columns = ['No_HP_Daftar', 'unique_nik_count']

        # Merge kembali ke dataframe utama
        df = df.merge(hp_nik_mapping, on='No_HP_Daftar', how='left')
        
        # ✅ PERBAIKAN: Handle NaN jika ada No_HP yang tidak ter-match
        df['unique_nik_count'] = df['unique_nik_count'].fillna(1).astype(int)

        # Rule Sindikat: Jika 1 HP dipakai > 15 NIK dalam dataset ini (proxy 30 hari)
        df['syndicate_score'] = 0
        df.loc[df['unique_nik_count'] > 15, 'syndicate_score'] = 40

        return df

    def train_and_predict_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Latih dan prediksi anomali menggunakan Isolation Forest"""
        features = ['distance_km', 'time_delta_hours', 'speed_kmh']
        X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

        if not self.is_fitted:
            self.iso_forest.fit(X)
            self.is_fitted = True

        # Score dari Isolation Forest
        raw_scores = self.iso_forest.decision_function(X)
        
        # Normalisasi ke 0-1 (semakin tinggi = semakin anomali)
        score_range = raw_scores.max() - raw_scores.min()
        if score_range > 0:
            df['anomaly_score'] = (raw_scores.max() - raw_scores) / score_range
        else:
            df['anomaly_score'] = 0.0

        return df

    def calculate_final_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gabungkan semua skor menjadi Risk Score 0-100 & Generate XAI"""
        df = df.copy()

        # Pastikan kolom numerik tidak NaN
        df['rule_penalty'] = df['rule_penalty'].fillna(0)
        df['syndicate_score'] = df['syndicate_score'].fillna(0)
        df['anomaly_score'] = df['anomaly_score'].fillna(0)

        # Bobot Skor
        df['risk_score'] = (
            df['rule_penalty'] +
            df['syndicate_score'] +
            (df['anomaly_score'] * 50)
        )

        # Normalize ke 0-100
        df['risk_score'] = df['risk_score'].clip(0, 100).round(2)

        # Generate Explainable AI (XAI) Reasons
        def generate_xai_reason(row):
            reasons = []
            if row['rule_penalty'] >= 50:
                reasons.append("Status kartu JKN Non-Aktif.")
            if row['speed_kmh'] > 100:
                reasons.append(
                    f"Anomali Waktu: Perpindahan {row['distance_km']:.0f}km "
                    f"dalam {row['time_delta_hours']:.1f} jam "
                    f"(Kecepatan: {row['speed_kmh']:.0f} km/h)."
                )
            if row['syndicate_score'] >= 40:
                reasons.append(
                    f"Indikasi Sindikat: No. HP terdaftar di "
                    f"{int(row['unique_nik_count'])} NIK berbeda."
                )
            return " | ".join(reasons) if reasons else "Klaim wajar."

        df['xai_reason'] = df.apply(generate_xai_reason, axis=1)

        # ✅ PERBAIKAN: Klasifikasi Risiko dengan handling edge case
        # Gunakan np.select agar tidak menghasilkan Categorical/NaN
        conditions = [
            df['risk_score'] <= 30,
            (df['risk_score'] > 30) & (df['risk_score'] <= 70),
            df['risk_score'] > 70
        ]
        choices = ['Aman (Hijau)', 'Review Ringan (Kuning)', 'Risiko Tinggi (Merah)']
        df['risk_category'] = np.select(conditions, choices, default='Aman (Hijau)')

        return df

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Eksekusi urutan pipeline"""
        df = self.feature_engineering(df)
        df = self.build_graph_syndicate(df)
        df = self.train_and_predict_anomaly(df)
        df = self.calculate_final_risk(df)
        return df