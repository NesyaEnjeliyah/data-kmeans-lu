import streamlit as st
import pandas as pd
import numpy as np
import os  # Tambahan untuk manajemen file database lokal

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# LOGO
# =====================================================
LOGO_PMS_URL   = "https://raw.githubusercontent.com/NesyaEnjeliyah/Analisa-KMeans-LayananUmum/c426c93032975bc37198a0479bdba9e3d624124f/dashboard/logo%20pms.png"
LOGO_UNAIR_URL = "https://raw.githubusercontent.com/NesyaEnjeliyah/Analisa-KMeans-LayananUmum/c426c93032975bc37198a0479bdba9e3d624124f/dashboard/Logo-Branding-UNAIR-biru.png"

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Dashboard Analisis Biaya Operasional",
    page_icon="⚓",
    layout="wide",
)

# Folder penyimpanan database lokal
DB_FOLDER = "database_operasional"

# =====================================================
# TEMA MARITIM BARU (PREMIUM RE-THEME) – CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@600;700&display=swap');

:root {
    --navy-deep:     #0C2C55;
    --ocean-teal:    #296374;
    --coastal-sky:   #629FAD;
    --sandy-cream:   #EDEDCE;
    --white:         #FFFFFF;
    --bg-soft:       #F9F9F4;
    --text-primary:  #0C2C55;
    --text-muted:    #296374;
    --border:        rgba(41, 99, 116, 0.22);
    --card-bg:       rgba(255, 255, 255, 0.95);
    --card-shadow:   0 4px 20px rgba(12, 44, 85, 0.06);
}

html, body {
    background-color: var(--bg-soft) !important;
    font-family: 'Barlow', sans-serif;
    color: var(--text-primary);
}

/* Siluet kapal ikonik interaktif pada latar belakang */
[data-testid="stAppViewContainer"] {
    background:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1440' height='900' viewBox='0 0 1440 900'%3E%3Cdefs%3E%3ClinearGradient id='sky' x1='0' y1='0' x2='0' y2='1'%3E%3Cstop offset='0%25' stop-color='%23F6F5ED'/%3E%3Cstop offset='100%25' stop-color='%23EDEDCE'/%3E%3C/linearGradient%3E%3ClinearGradient id='sea' x1='0' y1='0' x2='0' y2='1'%3E%3Cstop offset='0%25' stop-color='%23296374' stop-opacity='0.12'/%3E%3Cstop offset='100%25' stop-color='%230C2C55' stop-opacity='0.22'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='1440' height='900' fill='url(%23sky)'/%3E%3Cellipse cx='720' cy='820' rx='900' ry='180' fill='url(%23sea)'/%3E%3Cpath d='M0 640 Q180 610 360 640 Q540 670 720 640 Q900 610 1080 640 Q1260 670 1440 640' stroke='%23296374' stroke-width='2' fill='none' stroke-opacity='0.12'/%3E%3Cpath d='M0 660 Q180 630 360 660 Q540 690 720 660 Q900 630 1080 660 Q1260 690 1440 660' stroke='%23296374' stroke-width='1.5' fill='none' stroke-opacity='0.09'/%3E%3Cpath d='M0 680 Q180 650 360 680 Q540 710 720 680 Q900 650 1080 680 Q1260 710 1440 680' stroke='%23296374' stroke-width='1' fill='none' stroke-opacity='0.07'/%3E%3Cg transform='translate(980,530) scale(1.1)'%3E%3Crect x='0' y='40' width='200' height='38' rx='4' fill='%230C2C55' opacity='0.28'/%3E%3Crect x='10' y='20' width='130' height='24' rx='3' fill='%23296374' opacity='0.22'/%3E%3Crect x='30' y='8' width='50' height='14' rx='2' fill='%230C2C55' opacity='0.18'/%3E%3Crect x='40' y='0' width='4' height='10' fill='%23629FAD' opacity='0.22'/%3E%3Crect x='150' y='24' width='20' height='16' rx='2' fill='%230C2C55' opacity='0.18'/%3E%3Cellipse cx='100' cy='78' rx='100' ry='8' fill='%230C2C55' opacity='0.09'/%3E%3C/g%3E%3Cg transform='translate(60,570) scale(0.75)'%3E%3Crect x='0' y='36' width='160' height='30' rx='4' fill='%23296374' opacity='0.20'/%3E%3Crect x='10' y='18' width='100' height='20' rx='3' fill='%230C2C55' opacity='0.16'/%3E%3Crect x='20' y='6' width='40' height='14' rx='2' fill='%230C2C55' opacity='0.13'/%3E%3Crect x='28' y='0' width='3' height='8' fill='%23629FAD' opacity='0.18'/%3E%3Cellipse cx='80' cy='66' rx='80' ry='6' fill='%230C2C55' opacity='0.07'/%3E%3C/g%3E%3Cellipse cx='200' cy='120' rx='80' ry='28' fill='white' opacity='0.55'/%3E%3Cellipse cx='260' cy='108' rx='60' ry='22' fill='white' opacity='0.45'/%3E%3Cellipse cx='1100' cy='90' rx='100' ry='32' fill='white' opacity='0.50'/%3E%3Cellipse cx='1180' cy='78' rx='70' ry='24' fill='white' opacity='0.40'/%3E%3Cellipse cx='600' cy='150' rx='60' ry='20' fill='white' opacity='0.35'/%3E%3C/svg%3E") no-repeat center top / cover,
        linear-gradient(180deg, #F6F5ED 0%, #F9F9F4 50%, #EDEDCE 100%) !important;
    background-attachment: fixed !important;
}

[data-testid="stMain"] { background: transparent !important; }

.maritime-hero {
    background: linear-gradient(135deg, var(--navy-deep) 0%, var(--ocean-teal) 60%, var(--coastal-sky) 100%);
    border-bottom: 4px solid var(--sandy-cream);
    padding: 2.8rem 2.5rem 2rem;
    margin: -1rem -1rem 2rem;
    position: relative;
    overflow: hidden;
    text-align: center;
    box-shadow: 0 4px 24px rgba(12, 44, 85, 0.22);
}
.maritime-hero h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin: 0 auto;
    line-height: 1.1;
    text-shadow: 0 2px 12px rgba(0,0,0,0.3);
    max-width: 900px;
}
.maritime-hero .subtitle {
    font-size: 1rem;
    color: var(--sandy-cream);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    font-weight: 500;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 3px;
    padding: 0.18rem 0.8rem;
    font-size: 0.78rem;
    color: var(--sandy-cream);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
    font-weight: 600;
}
.wave-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(237,237,206,0.3), rgba(237,237,206,0.7), rgba(237,237,206,0.3), transparent);
    margin: 1.2rem auto 0;
    max-width: 500px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0C2C55 !important;
    border-right: none !important;
    min-width: 230px !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 !important;
}

/* ── Sidebar nav buttons: flat menu item rata kiri ── */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: rgba(255,255,255,0.65) !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: center !important;
    display: flex !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    box-shadow: none !important;
    transition: background 0.15s ease !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] span,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] div {
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    color: inherit !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #FFFFFF !important;
    transform: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover p,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover span {
    color: #FFFFFF !important;
}

/* Sembunyikan radio bawaan Streamlit, kita ganti dengan HTML custom */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    display: none !important;
}

/* Selector tahun di sidebar */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] svg {
    fill: rgba(255,255,255,0.6) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.10) !important;
}

/* Efek Elevasi Mikro-Interaksi UI/UX */
[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-top: 4px solid var(--ocean-teal) !important;
    border-radius: 10px !important;
    padding: 1.2rem 1.4rem !important;
    box-shadow: var(--card-shadow) !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 8px 24px rgba(12, 44, 85, 0.12) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--coastal-sky) !important;
}
[data-testid="stMetricValue"] {
    color: var(--navy-deep) !important;
}

h1, h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--navy-deep) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--ocean-teal), var(--navy-deep)) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    padding: 0.5rem 2rem !important;
    box-shadow: 0 3px 14px rgba(41, 99, 116, 0.25) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 20px rgba(41, 99, 116, 0.45) !important;
}

.year-pill {
    display: inline-block;
    background: rgba(41, 99, 116, 0.08);
    border: 1.5px solid var(--ocean-teal);
    border-radius: 20px;
    padding: 0.25rem 1rem;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--ocean-teal);
    letter-spacing: 0.08em;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

/* Pastikan kolom tengah tidak menggeser konten ke kiri */
[data-testid="stMain"] .stColumn {
    display: flex;
    flex-direction: column;
    align-items: stretch;
}

/* Override Streamlit default yang bisa membuat teks rata kiri */
[data-testid="stVerticalBlock"] > div {
    width: 100% !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed var(--ocean-teal) !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# PLOTLY TEMA PREMIUM
# =====================================================

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(237,237,206,0.15)",
    font=dict(family="Barlow, sans-serif", color="#296374", size=12),
    xaxis=dict(gridcolor="rgba(98,159,173,0.12)", linecolor="rgba(41,99,116,0.25)", tickcolor="rgba(41,99,116,0.25)"),
    yaxis=dict(gridcolor="rgba(98,159,173,0.12)", linecolor="rgba(41,99,116,0.25)", tickcolor="rgba(41,99,116,0.25)"),
)

TITLE_FONT_CONFIG = dict(family="Barlow Condensed, sans-serif", color="#0C2C55", size=16)
PALETTE_COLORS = ["#0C2C55", "#296374", "#629FAD", "#D6D6AF", "#8B5CF6", "#F97316", "#06B6D4"]

# =====================================================
# KONSTANTA MAPPING
# =====================================================

MAPPING_UNIT_KERJA = {
    "Operasi":                                          "Operasi",
    "Teknik":                                           "Teknik",
    "Keuangan":                                         "Keuangan",
    "Komersial":                                        "Komersial",
    "Sekretaris Perusahaan":                            "Sekretaris Perusahaan",
    "Satuan Pengawasan Internal":                       "Satuan Pengawasan Internal",
    "Kesisteman SDM dan Manajemen Risiko":              "Kesisteman SDM dan Manajemen Risiko",
    "Sistem Manajemen dan Teknologi Informasi":         "Sistem Manajemen dan Teknologi Informasi",
    "Pengadaan Barang dan Jasa dan Layanan Umum":       "PBJ dan Layanan Umum",
}

UNIT_DIHAPUS = [
    "Direktorat Keuangan, SDM dan Umum",
    "Direktorat Komersial, Operasi dan Teknik",
    "Direktorat Utama",
    "Komisaris",
]

MAPPING_KONSUMSI = {
    "Teknik / Engineering":                             "Teknik",
    "Operasi / Operations":                             "Operasi",
    "Sistem Manajemen dan Teknologi Informasi":         "Sistem Manajemen dan Teknologi Informasi",
    "Kesisteman SDM dan Manajemen Risiko":              "Kesisteman SDM dan Manajemen Risiko",
    "Komersial / Commercial":                           "Komersial",
    "Sekretaris Perusahaan / Corporate Secretary":      "Sekretaris Perusahaan",
    "Keuangan / Finance":                               "Keuangan",
    "PBJ dan Layanan Umum":                             "PBJ dan Layanan Umum",
    "Satuan Pengawas Intern (SPI)":                     "Satuan Pengawasan Internal",
}

# =====================================================
# DATABASE HELPER PERSISTENCE
# =====================================================

def load_local_database():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
        return {}
    
    database_tersimpan = {}
    for file in os.listdir(DB_FOLDER):
        if file.endswith("_sppd.xlsx"):
            tahun = file.split("_sppd.xlsx")[0]
            sppd_path = os.path.join(DB_FOLDER, f"{tahun}_sppd.xlsx")
            konsumsi_path = os.path.join(DB_FOLDER, f"{tahun}_konsumsi.xlsx")
            
            if os.path.exists(konsumsi_path):
                try:
                    database_tersimpan[tahun] = {
                        "sppd": pd.read_excel(sppd_path),
                        "konsumsi": pd.read_excel(konsumsi_path)
                    }
                except Exception as e:
                    st.error(f"Gagal memuat database lokal tahun {tahun}: {e}")
    return database_tersimpan

def save_to_local_database(tahun, sppd_df, konsumsi_df):
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    sppd_path = os.path.join(DB_FOLDER, f"{tahun}_sppd.xlsx")
    konsumsi_path = os.path.join(DB_FOLDER, f"{tahun}_konsumsi.xlsx")
    sppd_df.to_excel(sppd_path, index=False)
    konsumsi_df.to_excel(konsumsi_path, index=False)

def delete_from_local_database(tahun):
    sppd_path = os.path.join(DB_FOLDER, f"{tahun}_sppd.xlsx")
    konsumsi_path = os.path.join(DB_FOLDER, f"{tahun}_konsumsi.xlsx")
    if os.path.exists(sppd_path):
        os.remove(sppd_path)
    if os.path.exists(konsumsi_path):
        os.remove(konsumsi_path)

# =====================================================
# SESSION STATE INIT
# =====================================================

if "stage" not in st.session_state:
    st.session_state.stage = "home"

if "data_per_tahun" not in st.session_state:
    st.session_state.data_per_tahun = load_local_database()

if "tahun_aktif" not in st.session_state:
    st.session_state.tahun_aktif = None

if "dashboard" in st.session_state and st.session_state.dashboard:
    st.session_state.stage = "dashboard"

# =====================================================
# LOGOUT BUTTON
# =====================================================

if st.session_state.stage == "dashboard":
    col_logout = st.columns([7, 1])[1]
    with col_logout:
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.stage = "home"
            st.session_state.tahun_aktif = None
            st.rerun()

# =====================================================
# HELPER: PROCESS RAW DATA
# =====================================================

def process_data(sppd_raw, konsumsi_raw):
    sppd_awal = len(sppd_raw)
    sppd_setelah_dedup = sppd_raw.drop_duplicates(subset=["No. sppd", "Peserta"], keep="first") if sppd_awal > 0 else sppd_raw
    jumlah_duplikat_sppd = sppd_awal - len(sppd_setelah_dedup)
    sppd_setelah_hapus_noninti = sppd_setelah_dedup[~sppd_setelah_dedup["Unit Kerja"].isin(UNIT_DIHAPUS)] if len(sppd_setelah_dedup) > 0 else sppd_setelah_dedup
    jumlah_noninti_dihapus = len(sppd_setelah_dedup) - len(sppd_setelah_hapus_noninti)
    
    sppd = sppd_setelah_hapus_noninti.copy()
    if not sppd.empty and "Unit Kerja" in sppd.columns:
        sppd["Divisi"] = sppd["Unit Kerja"].map(MAPPING_UNIT_KERJA)
        sppd = sppd.dropna(subset=["Divisi"]).reset_index(drop=True)
    sppd_akhir = len(sppd)

    konsumsi_awal = len(konsumsi_raw)
    konsumsi = konsumsi_raw.copy()
    if not konsumsi.empty and "Departemen" in konsumsi.columns:
        konsumsi["Divisi"] = konsumsi["Departemen"].map(MAPPING_KONSUMSI)
        konsumsi = konsumsi.dropna(subset=["Divisi"]).reset_index(drop=True)
    konsumsi_akhir = len(konsumsi)
    jumlah_duplikat_konsumsi = konsumsi_raw.duplicated().sum() if konsumsi_awal > 0 else 0
    jumlah_missing_konsumsi  = konsumsi_raw["Jumlah Pegawai"].isnull().sum() if "Jumlah Pegawai" in konsumsi_raw.columns else 0

    if sppd_akhir > 0:
        sppd_agregat = (
            sppd.groupby("Divisi")
            .agg(
                Jumlah_SPPD                     = ("No. sppd",                  "count"),
                Total_Lama_Hari                 = ("Lama Hari",                  "sum"),
                Total_Realisasi_Uang_Saku       = ("Realisasi Uang Saku",        "sum"),
                Total_Realisasi_Bantuan_Lainnya = ("Realisasi Bantuan Lainnya",   "sum"),
                Total_Realisasi_Hotel           = ("Realisasi Hotel",             "sum"),
                Total_Realisasi_Transportasi    = ("Realisasi Transportasi",      "sum"),
            )
            .reset_index()
            .rename(columns={"Divisi": "Departemen"})
        )
    else:
        sppd_agregat = pd.DataFrame(columns=["Departemen", "Jumlah_SPPD", "Total_Lama_Hari", "Total_Realisasi_Uang_Saku", "Total_Realisasi_Bantuan_Lainnya", "Total_Realisasi_Hotel", "Total_Realisasi_Transportasi"])

    if konsumsi_akhir > 0:
        konsumsi_agregat = (
            konsumsi.groupby("Divisi")
            .agg(
                Jumlah_Transaksi_Konsumsi = ("No. Transaksi", "count"),
                Total_Pegawai_Konsumsi    = ("Jumlah Pegawai", "sum"),
                Total_Biaya_Konsumsi      = ("Jumlah (Rp)",    "sum"),
            )
            .reset_index()
            .rename(columns={"Divisi": "Departemen"})
        )
    else:
        konsumsi_agregat = pd.DataFrame(columns=["Departemen", "Jumlah_Transaksi_Konsumsi", "Total_Pegawai_Konsumsi", "Total_Biaya_Konsumsi"])

    gabungan = pd.merge(sppd_agregat, konsumsi_agregat, on="Departemen", how="outer").fillna(0)

    for col in gabungan.columns:
        if col != "Departemen":
            gabungan[col] = pd.to_numeric(gabungan[col], errors="coerce").fillna(0)

    meta = {
        "sppd_awal": sppd_awal,
        "sppd_akhir": sppd_akhir,
        "jumlah_duplikat_sppd": jumlah_duplikat_sppd,
        "jumlah_noninti_dihapus": jumlah_noninti_dihapus,
        "konsumsi_awal": konsumsi_awal,
        "konsumsi_akhir": konsumsi_akhir,
        "jumlah_duplikat_konsumsi": int(jumlah_duplikat_konsumsi),
        "jumlah_missing_konsumsi": jumlah_missing_konsumsi,
        "sppd_agregat": sppd_agregat,
        "konsumsi_agregat": konsumsi_agregat,
    }

    return gabungan, meta

def validasi_file(sppd_file, konsumsi_file):
    sppd_cek     = pd.read_excel(sppd_file)
    konsumsi_cek = pd.read_excel(konsumsi_file)

    kolom_sppd_wajib     = {"No. sppd", "Peserta", "Unit Kerja"}
    kolom_konsumsi_wajib = {"Departemen", "No. Transaksi"}

    sppd_valid     = kolom_sppd_wajib.issubset(set(sppd_cek.columns))
    konsumsi_valid = kolom_konsumsi_wajib.issubset(set(konsumsi_cek.columns))

    if not sppd_valid and not konsumsi_valid:
        return None, None, "⚠️ Kedua file tidak sesuai. Pastikan urutan upload sudah benar."
    elif not sppd_valid:
        return None, None, "⚠️ File 'Data SPPD' sepertinya bukan file SPPD yang valid."
    elif not konsumsi_valid:
        return None, None, "⚠️ File 'Data Konsumsi' sepertinya bukan file Konsumsi yang valid."

    return sppd_cek, konsumsi_cek, None

# =====================================================
# STAGE: HOME
# =====================================================

if st.session_state.stage == "home":

    # Pusatkan seluruh konten home ke tengah layar
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:

        # ── Header: Logo + Judul ──
        st.markdown(f"""
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <!-- Baris Logo -->
            <div style="display:flex; align-items:center; justify-content:center; gap:2rem; margin-bottom:1.4rem;">
                <img src="{LOGO_PMS_URL}" style="height:56px; object-fit:contain;" alt="Logo PMS">
                <img src="{LOGO_UNAIR_URL}" style="height:70px; object-fit:contain;" alt="Logo UNAIR">
            </div>
            <!-- Nama Instansi -->
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.9rem; letter-spacing:0.18em;
                text-transform:uppercase; color:var(--ocean-teal); margin-bottom:0.35rem; font-weight:600;">
                PT Pelindo Marine Service
            </div>
            <!-- Judul Dashboard -->
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.75rem; letter-spacing:0.10em;
                font-weight:700; text-transform:uppercase; color:var(--navy-deep); margin-bottom:2rem;
                line-height:1.15;">
                Dashboard Analisis Biaya Operasional
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.data_per_tahun:
            tahun_list = sorted(st.session_state.data_per_tahun.keys(), reverse=True)

            st.markdown("""
            <div style="color:var(--ocean-teal); font-size:1rem; letter-spacing:0.03em; margin-bottom:0.6rem;">
                Pilih tahun yang tersimpan di database atau masukkan tahun baru untuk memulai
            </div>
            <div style="background:rgba(255,255,255,0.92); border:1px solid var(--border);
                border-left:4px solid var(--ocean-teal); border-radius:10px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.25rem; letter-spacing:0.14em;
                    text-transform:uppercase; color:var(--ocean-teal); font-weight:700;">
                    ✅ Database File Terdeteksi (Tersimpan Lokal)
                </div>
            </div>
            """, unsafe_allow_html=True)

            tahun_pilih = st.selectbox("📂 Pilih tahun untuk dibuka:", tahun_list, key="tahun_pilih_home")

            if st.button("📊 Buka Dashboard", use_container_width=True, key="btn_buka"):
                st.session_state.tahun_aktif = tahun_pilih
                st.session_state.stage = "dashboard"
                st.rerun()

            st.markdown("<div style='text-align:center; color:var(--ocean-teal); font-size:0.82rem; margin:0.8rem 0;'>— atau masukkan tahun baru di bawah —</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(255,255,255,0.92); border:1px solid var(--border);
            border-left:4px solid var(--coastal-sky); border-radius:10px; padding:1.2rem 1.5rem; margin-bottom:0.5rem;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.25rem; letter-spacing:0.14em;
                text-transform:uppercase; color:var(--ocean-teal); font-weight:700;">
                📅 Input Tahun Baru
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Inisialisasi state validasi tahun ──
        if "tahun_warning" not in st.session_state:
            st.session_state.tahun_warning = ""
        if "tahun_is_valid" not in st.session_state:
            st.session_state.tahun_is_valid = False

        def validasi_tahun_realtime():
            val = st.session_state.get("tahun_baru_home", "").strip()
            if val == "":
                st.session_state.tahun_warning  = ""
                st.session_state.tahun_is_valid = False
            elif not val.isdigit():
                st.session_state.tahun_warning  = "⚠️ Tahun hanya boleh berisi angka, tanpa huruf atau simbol."
                st.session_state.tahun_is_valid = False
            elif len(val) < 4:
                st.session_state.tahun_warning  = f"⚠️ Tahun harus 4 digit — baru {len(val)} digit dimasukkan."
                st.session_state.tahun_is_valid = False
            elif len(val) > 4:
                st.session_state.tahun_warning  = f"⚠️ Tahun tidak boleh lebih dari 4 digit."
                st.session_state.tahun_is_valid = False
            elif int(val) < 2000 or int(val) > 2099:
                st.session_state.tahun_warning  = "⚠️ Masukkan tahun yang valid dalam rentang 2000 – 2099."
                st.session_state.tahun_is_valid = False
            elif val in st.session_state.data_per_tahun:
                st.session_state.tahun_warning  = f"⚠️ Data tahun {val} sudah tersimpan. Buka melalui pilihan di atas."
                st.session_state.tahun_is_valid = False
            else:
                st.session_state.tahun_warning  = ""
                st.session_state.tahun_is_valid = True

        st.text_input(
            "Tahun Data (4 digit):",
            placeholder="Contoh: 2025",
            key="tahun_baru_home",
            on_change=validasi_tahun_realtime,
        )

        # Tampilkan warning/sukses real-time di bawah input
        if st.session_state.tahun_warning:
            st.warning(st.session_state.tahun_warning)
        elif st.session_state.tahun_is_valid:
            val_preview = st.session_state.get("tahun_baru_home", "").strip()
            st.success(f"✅ Format tahun **{val_preview}** valid. Klik tombol di bawah untuk melanjutkan.")

        if st.button("➡️  Lanjut — Upload Data", use_container_width=True, key="btn_lanjut"):
            tahun_clean = st.session_state.get("tahun_baru_home", "").strip()
            if not st.session_state.tahun_is_valid:
                st.error("⚠️ Perbaiki input tahun terlebih dahulu sebelum melanjutkan.")
            else:
                st.session_state.tahun_warning  = ""
                st.session_state.tahun_is_valid = False
                st.session_state.tahun_aktif    = tahun_clean
                st.session_state.stage          = "upload"
                st.rerun()

# =====================================================
# STAGE: UPLOAD
# =====================================================

elif st.session_state.stage == "upload":
    tahun_aktif_upload = st.session_state.tahun_aktif

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown(f"""
        <div style="text-align:center; padding:1.5rem 0 0.5rem;">
            <!-- Logo baris -->
            <div style="display:flex; align-items:center; justify-content:center; gap:1.5rem; margin-bottom:1.2rem;">
                <img src="{LOGO_PMS_URL}"
                     style="height:44px; object-fit:contain;" alt="Logo Pelindo Marine Service">
                <div style="width:1px; height:40px; background:rgba(41,99,116,0.25);"></div>
                <img src="{LOGO_UNAIR_URL}"
                     style="height:44px; object-fit:contain;" alt="Logo Universitas Airlangga">
            </div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; letter-spacing:0.12em;
                text-transform:uppercase; color:var(--ocean-teal); margin-bottom:0.4rem;">Upload Data Tahun</div>
            <div style="display:inline-block; background:var(--navy-deep); color:white; border-radius:4px;
                padding:0.2rem 1rem; font-family:'Barlow Condensed',sans-serif; font-size:1.4rem;
                font-weight:700; letter-spacing:0.1em; margin-bottom:0.8rem;">{tahun_aktif_upload}</div>
            <div style="color:var(--text-muted); font-size:0.85rem; margin-bottom:1.5rem;">
                Unggah kedua file untuk melanjutkan dan menyimpannya ke database
            </div>
        </div>
        """, unsafe_allow_html=True)

        sppd_file     = st.file_uploader("📋  Data SPPD (.xlsx)", type=["xlsx"], key="uploader_sppd_v2")
        konsumsi_file = st.file_uploader("🍽️  Data Konsumsi Pegawai (.xlsx)", type=["xlsx"], key="uploader_konsumsi_v2")

        col_back, col_start = st.columns([1, 2])
        with col_back:
            if st.button("← Kembali", key="btn_back_upload"):
                st.session_state.stage = "home"
                st.session_state.tahun_aktif = None
                st.rerun()

        with col_start:
            if sppd_file and konsumsi_file:
                if st.button("⚓  MULAI ANALISIS", use_container_width=True, key="btn_mulai"):
                    sppd_df, konsumsi_df, err = validasi_file(sppd_file, konsumsi_file)
                    if err:
                        st.error(err)
                    else:
                        save_to_local_database(tahun_aktif_upload, sppd_df, konsumsi_df)
                        st.session_state.data_per_tahun[tahun_aktif_upload] = {"sppd": sppd_df, "konsumsi": konsumsi_df}
                        st.session_state.stage = "dashboard"
                        st.rerun()
            else:
                st.button("⚓  MULAI ANALISIS", use_container_width=True, disabled=True, key="btn_mulai_dis")
                st.caption("Unggah kedua file untuk mengaktifkan tombol ini.")

# =====================================================
# STAGE: DASHBOARD
# =====================================================

elif st.session_state.stage == "dashboard":
    tahun_tersedia = sorted(st.session_state.data_per_tahun.keys(), reverse=True)

    if st.session_state.tahun_aktif not in tahun_tersedia:
        st.session_state.tahun_aktif = tahun_tersedia[0] if tahun_tersedia else None

    if not st.session_state.tahun_aktif:
        st.error("Tidak ada data tersimpan. Silakan kembali ke halaman utama.")
        if st.button("← Ke Halaman Utama"):
            st.session_state.stage = "home"
            st.rerun()
        st.stop()

    # ── Sidebar: Header Brand ──
    st.sidebar.markdown("""
    <div style="padding:1.6rem 1.4rem 1.2rem; border-bottom:1px solid rgba(255,255,255,0.10);">
        <div style="display:flex; align-items:center; gap:0.7rem;">
            <div style="font-size:1.6rem; line-height:1;">⚓</div>
            <div>
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.05rem;
                    font-weight:700; color:#FFFFFF; letter-spacing:0.06em; text-transform:uppercase;
                    line-height:1.1;">Pelindo Marine</div>
                <div style="font-size:0.65rem; letter-spacing:0.14em; text-transform:uppercase;
                    color:rgba(237,237,206,0.65); margin-top:0.1rem;">Analisis Biaya Operasional</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gunakan session state untuk tracking menu aktif
    if "active_menu" not in st.session_state:
        st.session_state.active_menu = "Dashboard"

    st.sidebar.markdown("<div style='padding:0.8rem 0 0.2rem 1.2rem; font-size:0.62rem; letter-spacing:0.16em; text-transform:uppercase; color:rgba(237,237,206,0.50); font-weight:600;'>Menu Utama</div>", unsafe_allow_html=True)

    menu_icons = {
        "Dashboard":             "🗺️  Dashboard",
        "Data per Tahun":        "🗄️  Data per Tahun",
        "Pra-Pemrosesan":        "🔧  Pra-Pemrosesan",
        "K-Means Clustering":    "🔬  K-Means Clustering",
        "Analisis Pareto":       "📊  Analisis Pareto",
        "Pola Penggunaan Biaya": "🌊  Pola Penggunaan Biaya",
    }

    for key, label in menu_icons.items():
        is_active = st.session_state.active_menu == key
        # Highlight aktif dengan border-left via wrapper div
        if is_active:
            st.sidebar.markdown(f"""<div style="border-left:4px solid #629FAD; background:rgba(255,255,255,0.10);
                margin:0; padding:0;"></div>""", unsafe_allow_html=True)
        if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.active_menu = key
            st.rerun()

    menu = st.session_state.active_menu

    st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.10); margin:0.8rem 0 0.5rem'>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='padding:0 0 0.4rem 1.2rem; font-size:0.62rem; letter-spacing:0.16em; text-transform:uppercase; color:rgba(237,237,206,0.50); font-weight:600;'>Tahun Aktif</div>", unsafe_allow_html=True)

    if len(tahun_tersedia) > 1:
        tahun_aktif_new = st.sidebar.selectbox("", tahun_tersedia, index=tahun_tersedia.index(st.session_state.tahun_aktif), label_visibility="collapsed", key="year_selector_sidebar")
        if tahun_aktif_new != st.session_state.tahun_aktif:
            st.session_state.tahun_aktif = tahun_aktif_new
            st.rerun()
    else:
        st.sidebar.markdown(f"<div style='padding:0.5rem 1.2rem; font-family:Barlow Condensed,sans-serif; font-size:1.1rem; font-weight:700; color:#FFFFFF; letter-spacing:0.08em;'>{st.session_state.tahun_aktif}</div>", unsafe_allow_html=True)

    tahun_aktif = st.session_state.tahun_aktif
    sppd_raw     = st.session_state.data_per_tahun[tahun_aktif]["sppd"].copy()
    konsumsi_raw = st.session_state.data_per_tahun[tahun_aktif]["konsumsi"].copy()

    gabungan, meta = process_data(sppd_raw, konsumsi_raw)
    fitur = [c for c in gabungan.columns if c != "Departemen"]

    sppd_awal               = meta["sppd_awal"]
    sppd_akhir              = meta["sppd_akhir"]
    jumlah_duplikat_sppd    = meta["jumlah_duplikat_sppd"]
    jumlah_noninti_dihapus  = meta["jumlah_noninti_dihapus"]
    konsumsi_awal           = meta["konsumsi_awal"]
    konsumsi_akhir          = meta["konsumsi_akhir"]
    jumlah_duplikat_konsumsi= meta["jumlah_duplikat_konsumsi"]
    sppd_agregat            = meta["sppd_agregat"]
    konsumsi_agregat        = meta["konsumsi_agregat"]

    st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.10); margin:0.8rem 0'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style="padding:0.9rem 1.2rem; margin:0 0.6rem; background:rgba(255,255,255,0.07);
        border:1px solid rgba(255,255,255,0.10); border-radius:6px; font-size:0.75rem;
        color:rgba(255,255,255,0.65); line-height:1.8;">
        <div style="color:#629FAD; font-weight:700; font-size:0.65rem; letter-spacing:0.12em;
            text-transform:uppercase; margin-bottom:0.5rem;">Status Data · {tahun_aktif}</div>
        <div>Departemen&nbsp;&nbsp;<span style="color:#FFFFFF; font-weight:700; float:right;">{len(gabungan)}</span></div>
        <div>Baris SPPD&nbsp;&nbsp;<span style="color:#FFFFFF; font-weight:700; float:right;">{sppd_akhir:,}</span></div>
        <div>Konsumsi&nbsp;&nbsp;<span style="color:#FFFFFF; font-weight:700; float:right;">{konsumsi_akhir:,}</span></div>
        <div style="margin-top:0.5rem; color:rgba(237,237,206,0.45); font-size:0.65rem;">Total file tersimpan: {len(tahun_tersedia)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='display:inline-block; background:var(--navy-deep); color:white; border-radius:4px; padding:0.15rem 0.7rem; font-family:Barlow Condensed,sans-serif; font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase; font-weight:700; margin-bottom:0.8rem;'>📅 Data Aktif: {tahun_aktif}</div>", unsafe_allow_html=True)

    # =================================================
    # MENU: DASHBOARD
    # =================================================
    if menu == "Dashboard":
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; letter-spacing:0.18em; text-transform:uppercase; color:var(--ocean-teal); margin-bottom:0.8rem;">── Ringkasan Operasional</div>""", unsafe_allow_html=True)
        
        if gabungan.empty or len(gabungan) == 0:
            st.warning("⚠️ Data gabungan kosong. Tidak ada data operasional departemen.")
        else:
            total_biaya_sppd = gabungan["Total_Realisasi_Uang_Saku"].sum() + gabungan["Total_Realisasi_Bantuan_Lainnya"].sum() + gabungan["Total_Realisasi_Hotel"].sum() + gabungan["Total_Realisasi_Transportasi"].sum()
            total_biaya_konsumsi = gabungan["Total_Biaya_Konsumsi"].sum()
            total_biaya = total_biaya_sppd + total_biaya_konsumsi

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🏢 Departemen", len(gabungan))
            c2.metric("✈️ Total SPPD", int(gabungan["Jumlah_SPPD"].sum()))
            c3.metric("🍽️ Transaksi Konsumsi", int(gabungan["Jumlah_Transaksi_Konsumsi"].sum()))
            c4.metric("⚓ Total Biaya", f"Rp {total_biaya:,.0f}")

            chart_data = gabungan.copy()
            chart_data["Total_Biaya_SPPD"] = chart_data["Total_Realisasi_Uang_Saku"] + chart_data["Total_Realisasi_Bantuan_Lainnya"] + chart_data["Total_Realisasi_Hotel"] + chart_data["Total_Realisasi_Transportasi"]
            
            fig_bar = px.bar(chart_data.sort_values("Total_Biaya_SPPD", ascending=False), x="Departemen", y=["Total_Biaya_SPPD", "Total_Biaya_Konsumsi"], barmode="group", color_discrete_map={"Total_Biaya_SPPD": "#0C2C55", "Total_Biaya_Konsumsi": "#296374"})
            
            fig_bar.update_layout(
                **PLOTLY_THEME, 
                xaxis_tickangle=-30,
                title=dict(text="📊 Komposisi Distribusi Anggaran per Departemen", font=TITLE_FONT_CONFIG)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # =================================================
    # MENU: DATA PER TAHUN
    # =================================================
    elif menu == "Data per Tahun":
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; letter-spacing:0.18em; text-transform:uppercase; color:var(--ocean-teal); margin-bottom:0.3rem;">Arsip Data</div><h2>Data Operasional per Tahun</h2>""", unsafe_allow_html=True)
        for yr in sorted(tahun_tersedia, reverse=True):
            yr_sppd_raw, yr_konsumsi_raw = st.session_state.data_per_tahun[yr]["sppd"].copy(), st.session_state.data_per_tahun[yr]["konsumsi"].copy()
            yr_g, _ = process_data(yr_sppd_raw, yr_konsumsi_raw)
            yr_total = yr_g["Total_Realisasi_Uang_Saku"].sum() + yr_g["Total_Realisasi_Bantuan_Lainnya"].sum() + yr_g["Total_Realisasi_Hotel"].sum() + yr_g["Total_Realisasi_Transportasi"].sum() + yr_g["Total_Biaya_Konsumsi"].sum()
            active_style = "border-left: 4px solid var(--navy-deep);" if yr == tahun_aktif else "border-left: 4px solid var(--coastal-sky);"
            
            st.markdown(f"""<div style="{active_style} padding: 1rem; background: white; margin-bottom: 1rem; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);"><strong>📅 Tahun {yr}</strong> | 💰 Total Biaya: Rp {yr_total:,.0f}</div>""", unsafe_allow_html=True)
            with st.expander(f"▸ Lihat Tabel Gabungan Tahun {yr}"):
                st.dataframe(yr_g, use_container_width=True)
                if st.button(f"🗑️ Hapus Permanen {yr}", key=f"del_{yr}"):
                    delete_from_local_database(yr)
                    del st.session_state.data_per_tahun[yr]
                    st.rerun()

        st.markdown("<hr><h3>➕ Tambah Berkas Database Tahun Baru</h3>", unsafe_allow_html=True)
        tahun_baru_dash = st.text_input("📅 Tahun Data Baru:", placeholder="Contoh: 2026", key="tahun_baru_dashboard")
        sppd_baru  = st.file_uploader("📋 Data SPPD (.xlsx)", type=["xlsx"], key="sppd_baru_dash")
        kons_baru  = st.file_uploader("🍽️ Data Konsumsi (.xlsx)", type=["xlsx"], key="kons_baru_dash")
        
        if sppd_baru and kons_baru:
            if st.button("⚓ Simpan Data Baru", key="btn_simpan_baru"):
                t_clean = tahun_baru_dash.strip()
                if not t_clean.isdigit() or len(t_clean) != 4:
                    st.error("⚠️ Masukkan tahun 4 digit angka yang valid (Contoh: 2026).")
                elif t_clean in st.session_state.data_per_tahun:
                    st.warning(f"⚠️ Data tahun **{t_clean}** sudah terdaftar di database. Silakan hapus data lama tahun {t_clean} pada list berkas di atas.")
                else:
                    s_df, k_df, err2 = validasi_file(sppd_baru, kons_baru)
                    if err2:
                        st.error(err2)
                    else:
                        save_to_local_database(t_clean, s_df, k_df)
                        st.session_state.data_per_tahun[t_clean] = {"sppd": s_df, "konsumsi": k_df}
                        st.session_state.tahun_aktif = t_clean
                        st.success(f"✅ Data tahun {t_clean} sukses disimpan permanen!")
                        st.rerun()

    # =================================================
    # MENU: PRA-PEMROSESAN
    # =================================================
    elif menu == "Pra-Pemrosesan":
        st.markdown("<h2>Penanganan Missing Values & Agregasi</h2>", unsafe_allow_html=True)
        if gabungan.empty:
            st.warning("⚠️ Tabel agregasi kosong. Harap periksa apakah file Excel berisi data departemen.")
        else:
            st.markdown("##### Tabel Gabungan SPPD + Konsumsi")
            st.dataframe(gabungan, use_container_width=True)
            
            means, stds = gabungan[fitur].mean(), gabungan[fitur].std(ddof=1).replace(0, 1)
            zscore_df = pd.DataFrame(((gabungan[fitur] - means) / stds).values, columns=fitur)
            zscore_df.insert(0, "Departemen", gabungan["Departemen"].values)
            st.markdown("##### Hasil Normalisasi Z-Score")
            st.dataframe(zscore_df, use_container_width=True)

    # =================================================
    # MENU: K-MEANS CLUSTERING
    # =================================================
    elif menu == "K-Means Clustering":
        if len(gabungan) < 2:
            st.warning("⚠️ Tidak dapat menjalankan K-Means Clustering karena jumlah departemen aktif kurang dari 2.")
        else:
            means = gabungan[fitur].mean()
            stds  = gabungan[fitur].std(ddof=1).replace(0, 1)
            X_scaled = ((gabungan[fitur] - means) / stds).values

            k_min = 2
            k_max = max(k_min, min(7, len(gabungan) - 1))
            
            if k_max <= k_min:
                st.info("💡 Jumlah departemen aktif terlalu sedikit untuk grafik elbow.")
                k_pilihan = k_min
                model = KMeans(n_clusters=k_pilihan, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                hasil = gabungan.copy(); hasil["Cluster"] = "Cluster " + labels.astype(str)
                st.dataframe(hasil[["Departemen", "Cluster"] + fitur], use_container_width=True)
            else:
                K = range(k_min, k_max + 1)
                col_a, col_b = st.columns(2)
                with col_a:
                    inertia = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in K]
                    fig_elbow = px.line(x=list(K), y=inertia, markers=True, labels={"x":"k", "y":"Inertia"})
                    fig_elbow.update_traces(line=dict(color="#296374", width=2.5), marker=dict(color="#0C2C55", size=8))
                    fig_elbow.update_layout(**PLOTLY_THEME, title=dict(text="📈 Elbow Method (Evaluasi Optimal K)", font=TITLE_FONT_CONFIG))
                    st.plotly_chart(fig_elbow, use_container_width=True)
                with col_b:
                    scores = [silhouette_score(X_scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)) for k in K]
                    fig_sil = px.line(x=list(K), y=scores, markers=True, labels={"x":"k", "y":"Silhouette"})
                    fig_sil.update_traces(line=dict(color="#629FAD", width=2.5), marker=dict(color="#0C2C55", size=8))
                    fig_sil.update_layout(**PLOTLY_THEME, title=dict(text="✨ Silhouette Score (Kepadatan Klaster)", font=TITLE_FONT_CONFIG))
                    st.plotly_chart(fig_sil, use_container_width=True)

                best_k = int(np.argmax(scores)) + 2
                st.success(f"⚓ Cluster Optimal (Silhouette Score): **k = {best_k}**")

                # ── Inisialisasi state validasi K ──
                if "k_warning" not in st.session_state:
                    st.session_state.k_warning = ""
                if "k_valid" not in st.session_state:
                    st.session_state.k_valid = None

                def validasi_k_realtime():
                    val = st.session_state.get("k_input_realtime", "").strip()
                    if val == "":
                        st.session_state.k_warning = ""
                        st.session_state.k_valid   = None
                    elif "." in val or "," in val:
                        st.session_state.k_warning = "⚠️ Nilai k harus berupa angka bulat, bukan desimal."
                        st.session_state.k_valid   = None
                    else:
                        try:
                            k = int(val)
                            if k < 2 or k > 7:
                                st.session_state.k_warning = "⚠️ Nilai k harus berada dalam rentang 2 – 7."
                                st.session_state.k_valid   = None
                            elif k > k_max:
                                st.session_state.k_warning = f"⚠️ Maksimum k saat ini: {k_max} (jumlah departemen aktif – 1)."
                                st.session_state.k_valid   = None
                            else:
                                st.session_state.k_warning = ""
                                st.session_state.k_valid   = k
                        except ValueError:
                            st.session_state.k_warning = "⚠️ Masukan tidak valid. Harap masukkan angka bulat positif."
                            st.session_state.k_valid   = None

                st.text_input(
                    f"Masukkan nilai k (2 – {k_max}):",
                    value="",
                    placeholder=f"Contoh: {best_k}",
                    key="k_input_realtime",
                    on_change=validasi_k_realtime,
                )

                # Tampilkan feedback real-time
                if st.session_state.k_warning:
                    st.warning(st.session_state.k_warning)
                elif st.session_state.k_valid is not None:
                    st.success(f"✅ k = {st.session_state.k_valid} valid. Hasil clustering ditampilkan di bawah.")

                # Proses K-Means langsung begitu nilai valid
                if st.session_state.k_valid is not None:
                    k_pilihan = st.session_state.k_valid
                    model = KMeans(n_clusters=k_pilihan, random_state=42, n_init=10)
                    labels = model.fit_predict(X_scaled)
                    hasil = gabungan.copy(); hasil["Cluster"] = "Cluster " + labels.astype(str)

                    st.markdown("### Tabel Hasil Pembagian Cluster")
                    st.dataframe(hasil[["Departemen", "Cluster"] + fitur], use_container_width=True)

                    st.markdown("### Visualisasi Spasial Cluster — PCA 2D")
                    pca = PCA(n_components=2).fit_transform(X_scaled)
                    pca_df = pd.DataFrame({"PC1": pca[:, 0], "PC2": pca[:, 1], "Departemen": hasil["Departemen"].values, "Cluster": hasil["Cluster"].values})
                    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", text="Departemen", color_discrete_sequence=PALETTE_COLORS)
                    fig_pca.update_layout(**PLOTLY_THEME, title=dict(text="Pemetaan Klaster Departemen via PCA 2D", font=TITLE_FONT_CONFIG))
                    st.plotly_chart(fig_pca, use_container_width=True)

                    st.markdown("### Rata-rata Karakteristik Atribut per Cluster")
                    ringkasan = hasil.groupby("Cluster")[fitur].mean().reset_index()
                    st.dataframe(ringkasan, use_container_width=True)

                    st.markdown("### Analisis Profil Karakteristik Klaster")
                    hasil_analisis = hasil.copy()
                    hasil_analisis["Total_Biaya"] = hasil_analisis["Total_Realisasi_Uang_Saku"] + hasil_analisis["Total_Realisasi_Bantuan_Lainnya"] + hasil_analisis["Total_Realisasi_Hotel"] + hasil_analisis["Total_Realisasi_Transportasi"] + hasil_analisis["Total_Biaya_Konsumsi"]
                    rata_biaya_cluster = hasil_analisis.groupby("Cluster")["Total_Biaya"].mean().sort_values().reset_index()

                    kategori_cluster = {}
                    for rank, row in rata_biaya_cluster.iterrows():
                        if len(rata_biaya_cluster) == 2:
                            label, warna = ("Rendah", "#296374") if rank == 0 else ("Tinggi", "#0C2C55")
                        else:
                            if rank == 0: label, warna = "Rendah", "#629FAD"
                            elif rank == len(rata_biaya_cluster)-1: label, warna = "Tinggi", "#0C2C55"
                            else: label, warna = "Sedang", "#296374"
                        kategori_cluster[row["Cluster"]] = {"label": label, "warna": warna, "rata_biaya": row["Total_Biaya"]}

                    for c_nama in sorted(hasil_analisis["Cluster"].unique()):
                        info = kategori_cluster[c_nama]
                        dept_rows = hasil_analisis[hasil_analisis["Cluster"] == c_nama][["Departemen", "Total_Biaya"]].sort_values("Total_Biaya", ascending=False)
                        detail_dept = [f"{r['Departemen']} (Rp {r['Total_Biaya']:,.0f})" for _, r in dept_rows.iterrows()]
                        
                        st.markdown(f"""
                        <div style="border-left: 4px solid {info['warna']}; padding: 0.8rem; background: white; margin-bottom: 0.5rem; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.05);">
                            <strong>{c_nama} — Tingkat Penggunaan Biaya: <span style='color:{info['warna']}'>{info['label']}</span></strong><br>
                            <span style='font-size:0.85rem; color:#666;'>Rata-rata Pengeluaran Total: Rp {info['rata_biaya']:,.0f}</span><br>
                            <div style='margin-top:0.3rem; font-size:0.9rem;'>Anggota: {', '.join(detail_dept)}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # =================================================
    # MENU: ANALISIS PARETO
    # =================================================
    elif menu == "Analisis Pareto":
        st.markdown("<h2>Diagram Pareto Biaya Operasional</h2>", unsafe_allow_html=True)
        if gabungan.empty or len(gabungan) == 0:
            st.warning("⚠️ Tidak ada data untuk memetakan Diagram Pareto.")
        else:
            pareto = gabungan.copy()
            pareto["Total_Biaya"] = pareto["Total_Realisasi_Uang_Saku"] + pareto["Total_Realisasi_Bantuan_Lainnya"] + pareto["Total_Realisasi_Hotel"] + pareto["Total_Realisasi_Transportasi"] + pareto["Total_Biaya_Konsumsi"]
            pareto = pareto.sort_values("Total_Biaya", ascending=False).reset_index(drop=True)
            
            total_sum = pareto["Total_Biaya"].sum()
            pareto["Persentase (%)"] = (pareto["Total_Biaya"] / (total_sum if total_sum > 0 else 1) * 100).round(2)
            pareto["Kumulatif (%)"]  = pareto["Persentase (%)"].cumsum().round(2)

            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Bar(x=pareto["Departemen"], y=pareto["Total_Biaya"], name="Biaya (Rp)", yaxis="y1", marker=dict(color="#296374")))
            fig_pareto.add_trace(go.Scatter(x=pareto["Departemen"], y=pareto["Kumulatif (%)"], mode="lines+markers", name="Kumulatif (%)", yaxis="y2", line=dict(color="#0C2C55", width=2.5)))
            
            fig_pareto.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(237,237,206,0.15)", 
                title=dict(text="📊 Prioritas Pengeluaran Menggunakan Diagram Pareto 80/20", font=TITLE_FONT_CONFIG),
                yaxis=dict(title="Total Pengeluaran (Rp)"), 
                yaxis2=dict(title="Kumulatif (%)", overlaying="y", side="right", range=[0,110])
            )
            st.plotly_chart(fig_pareto, use_container_width=True)

            st.dataframe(pareto[["Departemen", "Total_Biaya", "Persentase (%)", "Kumulatif (%)"]], use_container_width=True)
            prioritas = pareto[pareto["Kumulatif (%)"] <= 85]
            if not prioritas.empty:
                st.success(f"⚓ **Fokus Utama Efisiensi:** Departemen ({', '.join(prioritas['Departemen'].tolist())}) berkontribusi paling dominan terhadap anggaran.")

    # =================================================
    # MENU: POLA PENGGUNAAN BIAYA
    # =================================================
    elif menu == "Pola Penggunaan Biaya":
        st.markdown("<h2>Heatmap Pola Penggunaan Biaya (Z-Score)</h2>", unsafe_allow_html=True)
        if gabungan.empty or len(gabungan) == 0:
            st.warning("⚠️ Tidak ada data untuk dianalisis pola sebarannya.")
        else:
            means, stds = gabungan[fitur].mean(), gabungan[fitur].std(ddof=1).replace(0, 1)
            zscore_df = pd.DataFrame(((gabungan[fitur] - means) / stds).values, columns=fitur, index=gabungan["Departemen"].values)

            fig_heatmap = px.imshow(zscore_df, text_auto=".2f", color_continuous_scale=[ [0.0, "#DB9891"], [0.25, "#EC5D4D"], [0.5, "#E3F2FD"], [0.75, "#FCA5A5"], [1.0, "#DC2626"]])
            fig_heatmap.update_layout(**PLOTLY_THEME, title=dict(text="🔥 Heatmap Distribusi Pola Anggaran (Z-Score)", font=TITLE_FONT_CONFIG))
            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.markdown("### Radar Chart — Profil Komparasi Departemen")
            dept_pilihan = st.selectbox("🔍 Pilih Departemen Untuk Dilihat Profilnya:", gabungan["Departemen"].tolist())
            nilai_z = zscore_df.loc[dept_pilihan].values.tolist()
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=nilai_z + [nilai_z[0]], theta=list(fitur) + [fitur[0]], fill="toself", name=dept_pilihan, fillcolor="rgba(41, 99, 116, 0.15)", line=dict(color="#0C2C55")))
            
            fig_radar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                polar=dict(bgcolor="rgba(237,237,206,0.15)"),
                title=dict(text=f"🧭 Profil Anggaran Radar: {dept_pilihan}", font=TITLE_FONT_CONFIG)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
