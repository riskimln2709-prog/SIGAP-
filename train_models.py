"""Melatih XGBoost + Isolation Forest, menyimpan model, dan melaporkan metrik pada data UJI (hold-out)."""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sigap_engine import SIGAPEngine, BACKEND, CFG


def metrics(y, pred):
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
    div = lambda a, b: round(a / b, 4) if b else None
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    f1 = round(2 * p * r / (p + r), 4) if p and r else None
    return dict(accuracy=div(tp + tn, len(y)), precision=p, recall=r, f1=f1, fpr=div(fp, fp + tn), tp=tp, fp=fp, fn=fn, tn=tn)


ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/inbox/klaim_harian.csv")
ap.add_argument("--out", default="data/out")
a = ap.parse_args()

df = pd.read_csv(a.data)
eng = SIGAPEngine(load=False)
feat = eng.build_features(df)
idx = np.arange(len(feat))
tr, te = train_test_split(idx, test_size=0.2, stratify=feat["label"], random_state=42)
eng.fit(feat.iloc[tr]); eng.save()

scored = eng.finalize(eng.score_ml(feat)).iloc[te]
y = scored["label"].astype(int)
rules_only = np.maximum.reduce([scored["rule_penalty"], scored["spatial_score"], scored["syndicate_score"]]) >= CFG["BATAS_MERAH"]
res = dict(backend=BACKEND, n_train=int(len(tr)), n_test=int(len(te)), fraud_rate_test=round(float(y.mean()), 4),
           merah=metrics(y, scored["risk_score"] >= CFG["BATAS_MERAH"]),
           merah_kuning=metrics(y, scored["risk_score"] >= CFG["BATAS_KUNING"]),
           hanya_aturan=metrics(y, pd.Series(rules_only, index=scored.index)),
           catatan="Dihitung pada 20% data uji sintetis yang tidak dipakai melatih model. Bukan performa pada data JKN asli.")
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
print(f"Model disimpan di models/ (backend: {BACKEND}). Data latih {len(tr)}, data uji {len(te)}.")
for k, t in [("Merah saja (skor >= 60)", "merah"), ("Merah + Kuning (skor >= 35)", "merah_kuning"), ("Hanya aturan, tanpa ML", "hanya_aturan")]:
    m = res[t]; print(f"{k:30s} recall={m['recall']}  presisi={m['precision']}  FPR={m['fpr']}  akurasi={m['accuracy']}")