import pandas as pd
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Path Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_DATA = os.path.join(BASE_DIR, 'data', 'X_FOODS_MENU.csv')

def siapkan_model():
    # 1. Load dataset menggunakan variabel path yang sudah dikonfigurasi
    df = pd.read_csv(PATH_DATA)
    
    # 2. Preprocessing Utama (Mempersiapkan teks untuk dibaca mesin)
    # KARENA KOLOM 'profil_rasa' SUDAH ADA DI CSV, kita tidak perlu menggabungkannya lagi.
    # Kita cukup menangani sel kosong dan mengubahnya menjadi huruf kecil
    df['profil_rasa'] = df['profil_rasa'].fillna('').str.lower()
    
    # 3. Penanganan informasi restoran yang kosong
    df['alamat'] = df['alamat'].fillna('Alamat tidak tersedia')
    df['link_restoran'] = df['link_restoran'].fillna('tidak tersedia')
    
    # 4. Normalisasi kategori
    df['kategori_clean'] = df['kategori_course'].str.lower().str.strip()
    
    # 5. Inisialisasi TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['profil_rasa'])
    
    return df, tfidf, tfidf_matrix

def cari_kombinasi_paket(pref_main, pref_bev, pref_side, budget_maksimal, df, tfidf, tfidf_matrix, jumlah_tampil=3):
    """Mencari kombinasi menu berdasarkan preferensi teks, batasan budget, dan keberagaman"""
    
    # 1. Ubah teks input menjadi matriks angka
    vec_main = tfidf.transform([pref_main])
    vec_bev = tfidf.transform([pref_bev])
    vec_side = tfidf.transform([pref_side])
    
    # 2. Hitung Cosine Similarity secara terpisah
    skor_main = cosine_similarity(vec_main, tfidf_matrix)[0]
    skor_bev = cosine_similarity(vec_bev, tfidf_matrix)[0]
    skor_side = cosine_similarity(vec_side, tfidf_matrix)[0]
    
    df_m = df.copy()
    df_b = df.copy()
    df_s = df.copy()
    
    df_m['skor_kemiripan'] = skor_main
    df_b['skor_kemiripan'] = skor_bev
    df_s['skor_kemiripan'] = skor_side
    
    # 3. Ambil 30 kandidat teratas (Diperbesar dari 15 agar menu langka punya ruang untuk dirakit)
    kandidat_main = df_m[df_m['kategori_course'] == 'Main Course'].sort_values('skor_kemiripan', ascending=False).head(30)
    kandidat_bev = df_b[df_b['kategori_course'] == 'Beverage'].sort_values('skor_kemiripan', ascending=False).head(30)
    # Ditambahkan 'Side Dish' ke dalam array isin untuk berjaga-jaga jika ada penyesuaian nama di dataset
    kandidat_side = df_s[df_s['kategori_course'].isin(['Appetizer', 'Dessert', 'Side Dish'])].sort_values('skor_kemiripan', ascending=False).head(30)
    
    semua_kombinasi = []
    
    # 4. Looping Kombinatorial dengan Batasan Budget
    for _, main in kandidat_main.iterrows():
        for _, bev in kandidat_bev.iterrows():
            for _, side in kandidat_side.iterrows():
                total_harga = main['harga_menu'] + bev['harga_menu'] + side['harga_menu']
                
                if total_harga <= budget_maksimal:
                    total_skor = main['skor_kemiripan'] + bev['skor_kemiripan'] + side['skor_kemiripan']
                    
                    # Konversi baris pandas ke dictionary agar aman untuk respon JSON FastAPI
                    semua_kombinasi.append({
                        'Main Course': main.to_dict(), 
                        'Beverage': bev.to_dict(), 
                        'Side Dish': side.to_dict(),
                        'Total Harga': int(total_harga), 
                        'Total Skor': float(total_skor)
                    })
                    
    # 5. Urutkan berdasarkan total skor tertinggi
    semua_kombinasi = sorted(semua_kombinasi, key=lambda x: x['Total Skor'], reverse=True)
    
    # 6. Susun Kolam Kandidat (Diperbesar dari 30 ke 150 kombinasi terbaik)
    kolam_kandidat = semua_kombinasi[:150]
    
    # 7. Acak kolam kandidat agar hasil dinamis (Random Sampling)
    random.shuffle(kolam_kandidat)
    
    paket_final = []
    main_terpilih = set()
    bev_terpilih = set()
    side_terpilih = set()
    
    # 8. FASE 1: Filter Keberagaman Ekstrem (Berusaha membuat 3 paket yang menunya beda total)
    for paket in kolam_kandidat:
        nama_main = paket['Main Course']['nama_menu']
        nama_bev = paket['Beverage']['nama_menu']
        nama_side = paket['Side Dish']['nama_menu']
        
        if nama_main not in main_terpilih and nama_bev not in bev_terpilih and nama_side not in side_terpilih:
            paket_final.append(paket)
            main_terpilih.add(nama_main)
            bev_terpilih.add(nama_bev)
            side_terpilih.add(nama_side)
            
        if len(paket_final) == jumlah_tampil:
            break
            
    # 9. FASE 2: Smart Fallback (Relaksasi Filter)
    # Jika kuota 3 paket belum terpenuhi (karena terhalang menu langka seperti 'Bubur'), 
    # turunkan syaratnya. Izinkan Main Course berulang, asalkan kombinasinya unik secara keseluruhan.
    if len(paket_final) < jumlah_tampil:
        for paket in kolam_kandidat:
            # Membuat ID unik untuk paket (Contoh: "Bubur Ayam - Es Teh - Puding")
            kombinasi_id = f"{paket['Main Course']['nama_menu']}-{paket['Beverage']['nama_menu']}-{paket['Side Dish']['nama_menu']}"
            terdaftar = [f"{p['Main Course']['nama_menu']}-{p['Beverage']['nama_menu']}-{p['Side Dish']['nama_menu']}" for p in paket_final]
            
            # Hanya tambahkan jika kombinasi utuh ini belum ada di paket_final
            if kombinasi_id not in terdaftar:
                paket_final.append(paket)
                
            if len(paket_final) == jumlah_tampil:
                break
                
    return paket_final
