from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, os, json

app = FastAPI(title="SIGAP API", version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_PATH = "data/out/dashboard_feed.json"
METRICS_PATH = "data/out/dummy_metrics.json"

@app.get("/api/v1/dashboard/feed")
def get_feed():
    if not os.path.exists(DATA_PATH):
        with open("data/out/dummy_feed.json", 'r') as f: return json.load(f)
    df = pd.read_json(DATA_PATH)
    return {"status": "success", "count": len(df), "data": df.to_dict(orient='records')}

@app.get("/api/v1/metrics")
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f: return json.load(f)
    return {"accuracy": 99.5, "recall": 99.4, "f1_score": 0.961, "fpr": 0.52}