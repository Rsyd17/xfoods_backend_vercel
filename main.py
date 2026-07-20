from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.mesin_rekomendasi import siapkan_model, cari_kombinasi_paket

app = FastAPI()

# Mengizinkan React (Frontend) untuk mengambil data dari FastAPI (Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://xfoods.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Menyiapkan mesin saat server pertama kali menyala
df, tfidf, tfidf_matrix = siapkan_model()

# Format kerangka data yang dikirim oleh React
class PreferensiUser(BaseModel):
    pref_main: str
    pref_bev: str
    pref_side: str
    budget: int

# Jalur API (Endpoint) untuk menerima request dari React
@app.post("/api/rekomendasi")
def get_rekomendasi(user_input: PreferensiUser):
    # Parameter jumlah_tampil diubah menjadi 3 sesuai kesepakatan
    hasil = cari_kombinasi_paket(
        user_input.pref_main, 
        user_input.pref_bev, 
        user_input.pref_side, 
        user_input.budget, 
        df, tfidf, tfidf_matrix, jumlah_tampil=3
    )
    return {"status": "success", "data": hasil}

@app.get("/api/statistik")
def get_statistik():
    return {
        "total_menu": len(df),
        "total_resto": df['nama_restoran'].nunique(),
        "harga_termurah": int(df['harga_menu'].min()),
        "harga_termahal": int(df['harga_menu'].max())
    }
def beranda():
    return{"pesan":"Mesin Rekomendasi Aktif dan Siap Menerima Data"}