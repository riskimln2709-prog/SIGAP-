"""
SIGAP Risk Engine - mengikuti Laporan Arsitektur v3.0 (+ aturan frekuensi dari Laporan Kapabilitas v1.2)

Lapis 1  Hard Rules      : kartu Non-Aktif -> 60 ; >=3 klaim/24 jam -> 40
Lapis 2  Spatiotemporal  : Haversine vektor; jarak >= 300 km dan jeda <= 3 jam -> 60
                           (dikecualikan: layanan Darurat, atau Rujukan ke RS Tipe A)
Lapis 3  Graph (NetworkX): 1 No. HP terhubung ke > 2 NIK -> 60
Lapis 4  ML Ensemble     : skor_ML = 0.7 * XGBoost(0-70) + 0.3 * IsolationForest(0-70)
Skor akhir = max(semua skor aturan, skor ML)   -> Merah >= 60, Kuning 35-59.99, Aman < 35
"""
import os
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import IsolationForest

try:
    from xgboost import XGBClassifier
    BACKEND = "xgboost"
except Exception:                       # cadangan jika xgboost belum terpasang
    from sklearn.ensemble import GradientBoostingClassifier
    BACKEND = "sklearn-gradient-boosting"

CFG = dict(
    RULE_NONAKTIF=60.0, RULE_FREKUENSI=40.0, FREKUENSI_N=3, FREKUENSI_JAM=24,
    JARAK_KM=300.0, JEDA_JAM=3.0, SKOR_SPASIAL=60.0,
    GRAF_DEGREE=2, SKOR_GRAF=60.0,
    ML_MAX=70.0, W_XGB=0.7, W_IF=0.3, IF_K=15.0,
    BATAS_MERAH=60.0, BATAS_KUNING=35.0,
)
DEFAULTS = {"Jenis_Layanan": "Rawat Jalan", "Tipe_RS": "B", "Biaya_Juta": np.nan}
FEAT_XGB = ["status_aktif_num", "jarak_km", "waktu_selisih_jam", "biaya_juta"]
FEAT_IF = ["biaya_juta", "waktu_selisih_jam"]
CAT_AMAN, CAT_KUNING, CAT_MERAH = "Aman (Hijau)", "Review Ringan (Kuning)", "Risiko Tinggi (Merah)"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class SIGAPEngine:
    def __init__(self, model_dir="models", load=True):
        self.model_dir = model_dir
        self.xgb = None
        self.iforest = None
        if load:
            self.load()

    # ---------- model ----------
    def _paths(self):
        return (os.path.join(self.model_dir, "model_sigap_xgb_v2.pkl"),
                os.path.join(self.model_dir, "model_sigap_iforest_v2.pkl"))

    def load(self):
        px, pi = self._paths()
        if os.path.exists(px) and os.path.exists(pi):
            self.xgb, self.iforest = joblib.load(px), joblib.load(pi)

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        px, pi = self._paths()
        joblib.dump(self.xgb, px)
        joblib.dump(self.iforest, pi)

    def fit(self, feat: pd.DataFrame):
        """Latih XGBoost (supervised, butuh kolom label) dan Isolation Forest (unsupervised)."""
        X, y = feat[FEAT_XGB], feat["label"].astype(int)
        if BACKEND == "xgboost":
            self.xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     eval_metric="logloss", random_state=42)
        else:
            self.xgb = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
        self.xgb.fit(X, y)
        self.iforest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        self.iforest.fit(feat[FEAT_IF])

    # ---------- Lapis 1-3 + fitur ----------
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, default in DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        df["No_HP_Daftar"] = df["No_HP_Daftar"].astype(str)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
        df = df.sort_values(["NIK_Hash", "Timestamp"]).reset_index(drop=True)
        med = df["Biaya_Juta"].median()
        df["biaya_juta"] = df["Biaya_Juta"].fillna(med if pd.notna(med) else 1.0)
        df["status_aktif_num"] = (df["Status_Aktif"] == "Aktif").astype(int)

        g = df.groupby("NIK_Hash")
        for c in ["Lat_Faskes", "Lng_Faskes", "Timestamp"]:
            df[c + "_1"] = g[c].shift(1)

        # Lapis 2: jarak (Haversine vektor) dan jeda waktu
        dist = haversine_km(df["Lat_Faskes_1"].to_numpy(float), df["Lng_Faskes_1"].to_numpy(float),
                            df["Lat_Faskes"].to_numpy(float), df["Lng_Faskes"].to_numpy(float))
        df["distance_km"] = np.nan_to_num(dist, nan=0.0)
        dt = (df["Timestamp"] - df["Timestamp_1"]).dt.total_seconds() / 3600
        df["time_delta_hours"] = dt.fillna(0.0)
        df["jarak_km"] = df["distance_km"]
        df["waktu_selisih_jam"] = dt.fillna(720.0).clip(upper=720.0)
        # klaim pertama NIK (tanpa pembanding) bernilai 0, bukan 9999
        speed = np.where(dt > 0, df["distance_km"] / dt.where(dt > 0, 1), np.where(df["distance_km"] > 0, 9999.0, 0.0))
        df["speed_kmh"] = np.where(dt.isna(), 0.0, speed)

        cond = (df["distance_km"] >= CFG["JARAK_KM"]) & (dt <= CFG["JEDA_JAM"])
        exempt = (df["Jenis_Layanan"] == "Darurat") | ((df["Jenis_Layanan"] == "Rujukan") & (df["Tipe_RS"] == "A"))
        df["spatial_score"] = np.where(cond & ~exempt, CFG["SKOR_SPASIAL"], 0.0)

        # Lapis 1: aturan keras
        n = CFG["FREKUENSI_N"]
        span = (df["Timestamp"] - g["Timestamp"].shift(n - 1)).dt.total_seconds() / 3600
        df["freq_flag"] = (span <= CFG["FREKUENSI_JAM"]).fillna(False)
        df["rule_penalty"] = np.maximum(np.where(df["Status_Aktif"] == "Non-Aktif", CFG["RULE_NONAKTIF"], 0.0),
                                        np.where(df["freq_flag"], CFG["RULE_FREKUENSI"], 0.0))

        # Lapis 3: graf NIK - No. HP
        edges = df[["No_HP_Daftar", "NIK_Hash"]].drop_duplicates()
        G = nx.Graph()
        G.add_edges_from((("hp", h), ("nik", k)) for h, k in edges.itertuples(index=False))
        deg = {h: G.degree(("hp", h)) for h in edges["No_HP_Daftar"].unique()}
        df["unique_nik_count"] = df["No_HP_Daftar"].map(deg).fillna(1).astype(int)
        df["syndicate_score"] = np.where(df["unique_nik_count"] > CFG["GRAF_DEGREE"], CFG["SKOR_GRAF"], 0.0)
        return df

    # ---------- Lapis 4: ML ----------
    def score_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.xgb is None or self.iforest is None:
            raise RuntimeError("Model belum dilatih. Jalankan dulu: python train_models.py")
        df = df.copy()
        mx = CFG["ML_MAX"]
        df["xgb_score"] = self.xgb.predict_proba(df[FEAT_XGB])[:, 1] * mx
        dec = self.iforest.decision_function(df[FEAT_IF])
        df["iforest_score"] = mx / (1.0 + np.exp(CFG["IF_K"] * dec))      # sigmoid terbalik
        df["ml_score"] = CFG["W_XGB"] * df["xgb_score"] + CFG["W_IF"] * df["iforest_score"]
        df["anomaly_score"] = df["iforest_score"] / mx                     # 0-1, kompatibel dengan dashboard lama
        return df

    # ---------- skor akhir + XAI ----------
    def finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["risk_score"] = np.maximum.reduce([df["rule_penalty"], df["spatial_score"],
                                              df["syndicate_score"], df["ml_score"]]).clip(0, 100).round(2)
        df["risk_category"] = np.select(
            [df["risk_score"] >= CFG["BATAS_MERAH"], df["risk_score"] >= CFG["BATAS_KUNING"]],
            [CAT_MERAH, CAT_KUNING], default=CAT_AMAN)

        def why(r):
            out = []
            if r["Status_Aktif"] == "Non-Aktif":
                out.append("Status Kartu Non-Aktif.")
            if r["freq_flag"]:
                out.append(f"Klaim berulang: {CFG['FREKUENSI_N']}+ klaim dalam {CFG['FREKUENSI_JAM']} jam.")
            if r["spatial_score"] > 0:
                out.append(f"Anomali Spatiotemporal ({r['distance_km']:.0f}km / {r['time_delta_hours']:.1f}jam).")
            if r["syndicate_score"] > 0:
                out.append(f"Indikasi Sindikat: No. HP terhubung ke {int(r['unique_nik_count'])} NIK berbeda.")
            if r["xgb_score"] >= CFG["BATAS_KUNING"]:
                out.append(f"Pola fraud terstruktur terdeteksi model (XGBoost, skor {r['xgb_score']:.0f}/70).")
            if r["iforest_score"] >= CFG["BATAS_KUNING"]:
                out.append("Anomali Pola Biaya/Durasi (IForest).")
            return " | ".join(out) if out else "Klaim wajar."

        df["xai_reason"] = df.apply(why, axis=1)
        return df

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.finalize(self.score_ml(self.build_features(df)))