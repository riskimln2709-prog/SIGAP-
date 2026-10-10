import pandas as pd
import numpy as np
import hashlib
import random
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

TOTAL_KLAIM = 10000
FRAUD_RATIO = 0.10
NUM_NORMAL = int(TOTAL_KLAIM * (1 - FRAUD_RATIO))
NUM_FRAUD = TOTAL_KLAIM - NUM_NORMAL

FASKES_DB = {
    "F-ACEH": {"lat": 5.5577, "lng": 95.3222}, "F-JKT1": {"lat": -6.2088, "lng": 106.8456},
    "F-JKT2": {"lat": -6.2254, "lng": 106.8515}, "F-BDG":  {"lat": -6.9175, "lng": 107.6191},
    "F-SBY":  {"lat": -7.2575, "lng": 112.7521}, "F-MDN":  {"lat": 3.5952,  "lng": 98.6722},
    "F-MKS":  {"lat": -5.1477, "lng": 119.4327}, "F-YOG":  {"lat": -7.7956, "lng": 110.3695},
}
ICD10_CODES = ["A09", "J00", "I10", "E11", "J18", "K29", "M54", "N39"]

def hash_nik(nik_raw): return hashlib.sha256(nik_raw.encode()).hexdigest()[:12]
def random_timestamp(base_date):
    days_offset = random.randint(0, 29)
    hour = random.randint(6, 20)
    dt = base_date - timedelta(days=days_offset, hours=-hour, minutes=-random.randint(0, 59))
    return dt.strftime("%Y-%m-%d %H:%M")
def add_noise(lat, lng): return round(lat + random.uniform(-0.04, 0.04), 4), round(lng + random.uniform(-0.04, 0.04), 4)

print(f"🟢 Generating {NUM_NORMAL} klaim NORMAL...")
normal_data, base_date = [], datetime(2026, 10, 8)
normal_patients = [{"NIK_Hash": hash_nik(f"3201{i:08d}"), "No_Kartu": f"000{random.randint(10000, 99999)}", 
                    "No_HP": f"08{random.choice(['12','13','57'])}{random.randint(10000000, 99999999)}", 
                    "Status_Aktif": "Aktif", "home_faskes": random.choice(list(FASKES_DB.keys()))} for i in range(2000)]

for i in range(NUM_NORMAL):
    p = random.choice(normal_patients)
    f = FASKES_DB[p["home_faskes"]]
    lat, lng = add_noise(f["lat"], f["lng"])
    normal_data.append({"ID_Klaim": f"K-N{i:05d}", "NIK_Hash": p["NIK_Hash"], "No_Kartu": p["No_Kartu"], 
                        "No_HP_Daftar": p["No_HP"], "Status_Aktif": p["Status_Aktif"], "Timestamp": random_timestamp(base_date), 
                        "Lat_Faskes": lat, "Lng_Faskes": lng, "Kode_Faskes": p["home_faskes"], "ICD-10": random.choice(ICD10_CODES), "label": 0})

print(f"🔴 Generating {NUM_FRAUD} klaim FRAUD...")
fraud_data, fraud_id = [], 0

# Modus 1: Impossible Travel (Aceha -> JKT in 2 hours)
for _ in range(int(NUM_FRAUD * 0.30) // 2):
    fraud_id += 1
    nik_hash, no_hp = hash_nik(f"9999{fraud_id:08d}"), f"0812{random.randint(10000000, 99999999)}"
    t1 = base_date - timedelta(days=random.randint(1, 28), hours=-8)
    t2 = t1 + timedelta(hours=2)
    for t, f_key in [(t1, "F-ACEH"), (t2, "F-JKT1")]:
        f = FASKES_DB[f_key]; lat, lng = add_noise(f["lat"], f["lng"])
        fraud_data.append({"ID_Klaim": f"K-F{fraud_id:05d}", "NIK_Hash": nik_hash, "No_Kartu": f"000{random.randint(10000, 99999)}", 
                           "No_HP_Daftar": no_hp, "Status_Aktif": "Aktif", "Timestamp": t.strftime("%Y-%m-%d %H:%M"), 
                           "Lat_Faskes": lat, "Lng_Faskes": lng, "Kode_Faskes": f_key, "ICD-10": random.choice(ICD10_CODES), "label": 1})

# Modus 2: Kartu Non-Aktif
for _ in range(int(NUM_FRAUD * 0.25)):
    fraud_id += 1; f_key = random.choice(list(FASKES_DB.keys())); f = FASKES_DB[f_key]; lat, lng = add_noise(f["lat"], f["lng"])
    fraud_data.append({"ID_Klaim": f"K-F{fraud_id:05d}", "NIK_Hash": hash_nik(f"8888{fraud_id:08d}"), "No_Kartu": f"000{random.randint(10000, 99999)}", 
                       "No_HP_Daftar": f"0813{random.randint(10000000, 99999999)}", "Status_Aktif": "Non-Aktif", "Timestamp": random_timestamp(base_date), 
                       "Lat_Faskes": lat, "Lng_Faskes": lng, "Kode_Faskes": f_key, "ICD-10": random.choice(ICD10_CODES), "label": 1})

# Modus 3: Sindikat 1 HP Banyak NIK
syndicate_hps = [f"0857{random.randint(10000000, 99999999)}" for _ in range(5)]
for hp in syndicate_hps:
    for j in range(20):
        if len(fraud_data) >= NUM_FRAUD: break
        fraud_id += 1; f_key = random.choice(["F-JKT1", "F-JKT2", "F-BDG"]); f = FASKES_DB[f_key]; lat, lng = add_noise(f["lat"], f["lng"])
        fraud_data.append({"ID_Klaim": f"K-F{fraud_id:05d}", "NIK_Hash": hash_nik(f"7777{fraud_id:08d}"), "No_Kartu": f"000{random.randint(10000, 99999)}", 
                           "No_HP_Daftar": hp, "Status_Aktif": "Aktif", "Timestamp": random_timestamp(base_date), 
                           "Lat_Faskes": lat, "Lng_Faskes": lng, "Kode_Faskes": f_key, "ICD-10": random.choice(ICD10_CODES), "label": 1})

# Gabung & Simpan
df_all = pd.concat([pd.DataFrame(normal_data), pd.DataFrame(fraud_data)], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
df_all['ID_Klaim'] = [f"K-{i+1:05d}" for i in range(len(df_all))]

os.makedirs("data/inbox", exist_ok=True)
df_all.to_csv("data/inbox/klaim_harian.csv", index=False)
print(f"✅ Data tersimpan di data/inbox/klaim_harian.csv (Total: {len(df_all)})")