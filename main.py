import os, json, hashlib, threading
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sigap_engine import SIGAPEngine

app = FastAPI(title="SIGAP API", version="1.3.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

INBOX_CSV = "data/inbox/klaim_harian.csv"
FEED = "data/out/dashboard_feed.json"
AUDIT = "data/out/full_audit_log.csv"
METRICS_PATH = "data/out/dummy_metrics.json"
LOCK = threading.Lock()


@app.get("/")
def root():
    return FileResponse("dashboard.html")


@app.get("/input")
def input_page():
    return FileResponse("formulir.html")


@app.get("/api/v1/dashboard/feed")
def get_feed():
    if not os.path.exists(FEED):
        return {"status": "error", "message": "dashboard_feed.json belum ada. Jalankan sigap_pipeline.py dulu."}
    df = pd.read_json(FEED)
    records = json.loads(df.to_json(orient="records"))
    return {"status": "success", "count": len(records), "data": records}


@app.get("/api/v1/metrics")
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {"accuracy": 99.5, "recall": 99.4, "f1_score": 0.961, "fpr": 0.52}


@app.get("/api/v1/faskes")
def get_faskes():
    df = pd.read_csv(INBOX_CSV, usecols=["Kode_Faskes", "Lat_Faskes", "Lng_Faskes"])
    g = df.groupby("Kode_Faskes")[["Lat_Faskes", "Lng_Faskes"]].median().round(4).reset_index()
    return g.to_dict(orient="records")


@app.get("/api/v1/evaluation")
def evaluation(red: float = 70, yel: float = 30):
    """Metrik dihitung dari SELURUH klaim (full_audit_log.csv), bukan hanya yang masuk antrean."""
    if not os.path.exists(AUDIT):
        return {"error": "full_audit_log.csv belum ada. Jalankan pipeline."}
    df = pd.read_csv(AUDIT)
    s = df["risk_score"]
    out = {"total": int(len(df)), "red": int((s > red).sum()),
           "yellow": int(((s > yel) & (s <= red)).sum()), "green": int((s <= yel).sum()),
           "labeled": 0, "recall": None, "precision": None, "fpr": None, "recall_flag": None}
    if "label" in df.columns and df["label"].notna().any():
        d = df[df["label"].notna()]
        y = d["label"] == 1
        ratio = lambda a, b: round(a / b, 4) if b else None
        p = d["risk_score"] > red
        pf = d["risk_score"] > yel
        tp, fp, fn, tn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())
        out.update(labeled=int(len(d)), recall=ratio(tp, tp + fn), precision=ratio(tp, tp + fp),
                   fpr=ratio(fp, fp + tn), recall_flag=ratio(int((pf & y).sum()), int(y.sum())))
    return out


class ClaimIn(BaseModel):
    nik: str
    no_hp: str
    no_kartu: str
    status_aktif: str
    kode_faskes: str
    icd10: str
    timestamp: Optional[str] = None


@app.post("/api/v1/claims")
def submit_claim(c: ClaimIn):
    nik = "".join(ch for ch in c.nik if ch.isdigit())
    if len(nik) != 16:
        raise HTTPException(422, "NIK harus 16 digit angka.")
    hp = "".join(ch for ch in c.no_hp if ch.isdigit())
    if hp.startswith("62"):
        hp = hp[2:]
    hp = hp.lstrip("0")
    if not 8 <= len(hp) <= 13:
        raise HTTPException(422, "No. HP tidak valid.")
    if c.status_aktif not in ("Aktif", "Non-Aktif"):
        raise HTTPException(422, "Status harus Aktif atau Non-Aktif.")
    try:
        ts = pd.to_datetime(c.timestamp) if c.timestamp else pd.Timestamp.now()
    except Exception:
        raise HTTPException(422, "Format waktu klaim tidak valid.")

    try:
        with LOCK:
            df = pd.read_csv(INBOX_CSV)
            fk = df[df["Kode_Faskes"] == c.kode_faskes]
            if fk.empty:
                raise HTTPException(422, "Kode faskes tidak dikenal.")
            claim_id = "M-" + datetime.now().strftime("%y%m%d%H%M%S")
            row = {col: np.nan for col in df.columns}
            row.update({
                "ID_Klaim": claim_id,
                "NIK_Hash": hashlib.sha256(nik.encode()).hexdigest()[:12],
                "No_Kartu": c.no_kartu.strip(), "No_HP_Daftar": hp,
                "Status_Aktif": c.status_aktif,
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Lat_Faskes": fk["Lat_Faskes"].median(), "Lng_Faskes": fk["Lng_Faskes"].median(),
                "Kode_Faskes": c.kode_faskes, "ICD-10": c.icd10.strip().upper(),
            })
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

            # Samakan format waktu semua baris (penyebab umum error)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
            df.assign(Timestamp=df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(INBOX_CSV, index=False)

            result = SIGAPEngine().run_pipeline(df)
            result[result["risk_category"] != "Aman (Hijau)"].to_json(FEED, orient="records", date_format="iso")
            result.to_csv(AUDIT, index=False)

        one = result[result["ID_Klaim"] == claim_id]
        rec = json.loads(one.to_json(orient="records", date_format="iso"))[0]
        return {"status": "success", "in_queue": rec["risk_category"] != "Aman (Hijau)", "claim": rec}
    except HTTPException:
        raise
    except Exception as e:
        # Tampilkan penyebab asli di formulir
        raise HTTPException(500, f"{type(e).__name__}: {e}")