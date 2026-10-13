"""Generator data klaim SINTETIS untuk SIGAP. Tidak memakai data JKN asli.
Pola fraud sengaja dibuat tidak selalu mudah (ada kasus samar dan kasus wajar yang mirip fraud)."""
import os
import numpy as np
import pandas as pd
from privacy import hash_nik

rng = np.random.RandomState(7)
TARGET = 10000
START = pd.Timestamp("2026-09-10")
DAYS = 29

# kode: (lat, lng, tipe RS)
FASKES = {
    "F-ACEH": (5.5483, 95.3238, "B"), "F-MDN": (3.5952, 98.6722, "A"), "F-PLG": (-2.9761, 104.7754, "B"),
    "F-JKT1": (-6.2259, 106.8590, "A"), "F-JKT2": (-6.1751, 106.8650, "B"), "F-BDG": (-6.9175, 107.6191, "B"),
    "F-SMG": (-6.9667, 110.4167, "B"), "F-YOG": (-7.7956, 110.3695, "A"), "F-SBY": (-7.2575, 112.7521, "A"),
    "F-DPS": (-8.6500, 115.2167, "B"), "F-MKS": (-5.1477, 119.4327, "A"), "F-MDO": (1.4748, 124.8421, "C"),
}
CODES = list(FASKES)
ICD = ["I10", "E11", "J18", "A09", "N39", "K29", "J06", "M54", "L20", "B34"]
rows = []  # dict per klaim
nik_seq = [0]


def rint(lo, hi):
    """Bilangan bulat acak yang aman di Windows (int64, bukan int32)."""
    return int(rng.randint(lo, hi, dtype=np.int64))


def new_nik():
    nik_seq[0] += 1
    return hash_nik("3174%012d" % (rint(0, 10**12) + nik_seq[0]))


def new_hp():
    return "8%d" % rint(10**9, 10**10 - 1)


def biaya(layanan, inflate=1.0):
    mu, sd, lo, hi = {"Rawat Jalan": (0.4, 0.5, 0.05, 3), "Rawat Inap": (4.0, 0.6, 0.8, 25), "Darurat": (3.0, 0.6, 0.5, 20),
                      "Rujukan": (1.5, 0.6, 0.2, 12)}[layanan]
    return round(float(np.clip(rng.lognormal(np.log(mu), sd), lo, hi)) * inflate, 3)


def layanan_acak():
    return rng.choice(["Rawat Jalan", "Rawat Inap", "Darurat"], p=[0.70, 0.25, 0.05])


def add(nik, hp, kartu, status, ts, fk, layanan, label, inflate=1.0):
    lat, lng, tipe = FASKES[fk]
    rows.append(dict(NIK_Hash=nik, No_Kartu=kartu, No_HP_Daftar=hp, Status_Aktif=status,
                     Timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"), Lat_Faskes=lat, Lng_Faskes=lng, Kode_Faskes=fk,
                     Tipe_RS=tipe, Jenis_Layanan=layanan, **{"ICD-10": rng.choice(ICD)},
                     Biaya_Juta=biaya(layanan, inflate), label=label))


def rand_ts(extra_h=0):
    return START + pd.Timedelta(hours=float(rng.uniform(0, DAYS * 24 - extra_h)))


def kartu():
    return str(rint(10**12, 10**13 - 1))


def far_pair():
    while True:
        a, b = rng.choice(CODES, 2, replace=False)
        la, lo, _ = FASKES[a]; lb, lb2, _ = FASKES[b]
        d = np.hypot((la - lb) * 111, (lo - lb2) * 111 * np.cos(np.radians(la)))
        if d >= 330:
            return a, b


# ---- FRAUD ----
for _ in range(150):                       # A. kartu Non-Aktif dipakai
    add(new_nik(), new_hp(), kartu(), "Non-Aktif", rand_ts(), rng.choice(CODES), layanan_acak(), 1)
for _ in range(120):                       # B. impossible travel jelas (<=3 jam)
    n, h, k = new_nik(), new_hp(), kartu(); a, b = far_pair(); t = rand_ts(4)
    add(n, h, k, "Aktif", t, a, "Rawat Jalan", 0)
    add(n, h, k, "Aktif", t + pd.Timedelta(hours=float(rng.uniform(0.5, 3))), b, "Rawat Jalan", 1)
for _ in range(60):                        # B2. perpindahan samar (3-6 jam); 20% ternyata wajar
    n, h, k = new_nik(), new_hp(), kartu(); a, b = far_pair(); t = rand_ts(7)
    add(n, h, k, "Aktif", t, a, "Rawat Jalan", 0)
    add(n, h, k, "Aktif", t + pd.Timedelta(hours=float(rng.uniform(3.2, 6))), b, "Rawat Jalan", int(rng.rand() > 0.2))
for _ in range(25):                        # C. sindikat: 1 HP, 4-20 NIK
    h = new_hp()
    for _ in range(rng.randint(4, 21)):
        add(new_nik(), h, kartu(), "Aktif", rand_ts(), rng.choice(CODES), layanan_acak(), 1)
for _ in range(60):                        # D. klaim berulang >=3 dalam 24 jam
    n, h, k = new_nik(), new_hp(), kartu(); t = rand_ts(30); fk = rng.choice(CODES)
    for j in range(4):
        add(n, h, k, "Aktif", t + pd.Timedelta(hours=float(rng.uniform(0, 20))), fk, "Rawat Jalan", int(j >= 2))
for _ in range(100):                       # E. penggelembungan biaya
    add(new_nik(), new_hp(), kartu(), "Aktif", rand_ts(), rng.choice(CODES), layanan_acak(), 1, inflate=rng.uniform(4, 8))

# ---- NORMAL ----
shared_pool = []
while len(rows) < TARGET:
    n, k = new_nik(), kartu()
    r = rng.rand()
    if r < 0.12 and shared_pool:           # keluarga berbagi HP (2 NIK)
        h = shared_pool.pop()
    else:
        h = new_hp()
        if r < 0.14:
            shared_pool.append(h)
    status = "Non-Aktif" if rng.rand() < 0.01 else "Aktif"   # status terlambat diperbarui (wajar)
    home = rng.choice(CODES); t = rand_ts(250)
    for j in range(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])):
        fk = home
        lay = layanan_acak()
        if rng.rand() < 0.08:              # berobat di kota lain (jeda >= 1 hari)
            fk = rng.choice(CODES)
            if FASKES[fk][2] == "A" and rng.rand() < 0.5:
                lay = "Rujukan"
        infl = rng.uniform(3, 5) if rng.rand() < 0.015 else 1.0   # kasus berat, biaya tinggi tapi wajar
        add(n, h, k, status, t, fk, lay, 0, infl)
        t = t + pd.Timedelta(hours=float(rng.uniform(26, 240)))
    if rng.rand() < 0.03:                  # dua klaim <=3 jam di faskes yang sama
        add(n, h, k, status, t + pd.Timedelta(hours=2), home, "Rawat Jalan", 0)
    if rng.rand() < 0.02:                  # darurat/rujukan RS-A jauh dalam <=3 jam (dikecualikan, wajar)
        a, b = far_pair(); tt = rand_ts(4)
        add(n, h, k, "Aktif", tt, a, "Rawat Jalan", 0)
        far_tipe = FASKES[b][2]
        add(n, h, k, "Aktif", tt + pd.Timedelta(hours=float(rng.uniform(1, 3))), b,
            "Darurat" if far_tipe != "A" else "Rujukan", 0)

df = pd.DataFrame(rows).sample(frac=1, random_state=3).reset_index(drop=True).iloc[:TARGET]
df.insert(0, "ID_Klaim", ["K-%05d" % (i + 1) for i in range(len(df))])
os.makedirs("data/inbox", exist_ok=True); os.makedirs("data/out", exist_ok=True)
df.to_csv("data/inbox/klaim_harian.csv", index=False)
print(f"Data tersimpan di data/inbox/klaim_harian.csv (Total: {len(df)}, fraud: {int(df.label.sum())} = {df.label.mean():.1%})")