import pandas as pd
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Path Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_DATA = os.path.join(BASE_DIR, 'data', 'X_FOODS_MENU_UPDATED.csv')

def siapkan_model():
    # 1. Load dataset menggunakan variabel path yang sudah dikonfigurasi
    df = pd.read_csv(PATH_DATA)
    
    # 2. Preprocessing Utama 
    df['profil_rasa'] = df['profil_rasa'].fillna('').str.lower()
    df['alamat'] = df['alamat'].fillna('Alamat tidak tersedia')
    df['link_restoran'] = df['link_restoran'].fillna('tidak tersedia')
    df['kategori_clean'] = df['kategori_course'].str.lower().str.strip()
    
    # 3. Inisialisasi TF-IDF
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
    
    # ==============================================================
    # 2.5. LOGIKA TAMBAHAN: KEYWORD BOOSTING & NORMALISASI
    # ==============================================================
    kata_abaikan = ['dan', 'atau', 'dengan', 'yang', 'untuk', 'dari', 'rasa']
    
    # Boosting untuk Main Course
    kata_kunci_main = [kata.strip() for kata in pref_main.lower().split()]
    for kata in kata_kunci_main:
        if len(kata) > 2 and kata not in kata_abaikan:
            mask_m = df_m['nama_menu'].str.lower().str.contains(kata, na=False) | \
                     df_m['profil_rasa'].str.contains(kata, na=False)
            df_m.loc[mask_m, 'skor_kemiripan'] += 0.3

    # NORMALISASI SKOR (Mengembalikan skor ke range 0.0 - 1.0 tanpa merusak hierarki)
    max_m = df_m['skor_kemiripan'].max()
    if max_m > 0:
        df_m['skor_kemiripan'] = df_m['skor_kemiripan'] / max_m
        
    max_b = df_b['skor_kemiripan'].max()
    if max_b > 0:
        df_b['skor_kemiripan'] = df_b['skor_kemiripan'] / max_b
        
    max_s = df_s['skor_kemiripan'].max()
    if max_s > 0:
        df_s['skor_kemiripan'] = df_s['skor_kemiripan'] / max_s
    # ==============================================================

    # ==============================================================
    # 3. AMBIL KANDIDAT TERATAS
    # Random sampling di awal DIHAPUS agar menu yang masuk budget 
    # tidak terbuang secara kebetulan. Kita gunakan to_dict('records') 
    # agar looping komputasi 10x lebih cepat.
    # ==============================================================
    kandidat_main = df_m[df_m['kategori_course'] == 'Main Course'].sort_values('skor_kemiripan', ascending=False).head(40).to_dict('records')
    kandidat_bev = df_b[df_b['kategori_course'] == 'Beverage'].sort_values('skor_kemiripan', ascending=False).head(30).to_dict('records')
    kandidat_side = df_s[df_s['kategori_course'].isin(['Appetizer', 'Dessert', 'Side Dish'])].sort_values('skor_kemiripan', ascending=False).head(30).to_dict('records')
    
    semua_kombinasi = []
    
    # 4. Looping Kombinatorial dengan Batasan Budget
    for main in kandidat_main:
        for bev in kandidat_bev:
            for side in kandidat_side:
                total_harga = main['harga_menu'] + bev['harga_menu'] + side['harga_menu']
                
                # Filter Budget
                if total_harga <= budget_maksimal:
                    total_skor = main['skor_kemiripan'] + bev['skor_kemiripan'] + side['skor_kemiripan']
                    
                    # Karena sudah pakai to_dict('records'), main, bev, side sudah berupa dictionary
                    semua_kombinasi.append({
                        'Main Course': main, 
                        'Beverage': bev, 
                        'Side Dish': side,
                        'Total Harga': int(total_harga), 
                        'Total Skor': float(total_skor)
                    })
                    
    # 5. Urutkan berdasarkan total skor tertinggi dari kombinasi yang VALID (Masuk Budget)
    semua_kombinasi = sorted(semua_kombinasi, key=lambda x: x['Total Skor'], reverse=True)
    
    # 6. Susun Kolam Kandidat (Diperbesar ke 300 agar variasi kocokan menu lebih kaya)
    kolam_kandidat = semua_kombinasi[:300]
    
    # 7. Acak kolam kandidat agar hasil dinamis (Random Sampling di tahap AKHIR)
    random.shuffle(kolam_kandidat)
    
    paket_final = []
    main_terpilih = set()
    bev_terpilih = set()
    side_terpilih = set()
    
    # 8. FASE 1: Filter Keberagaman Ekstrem
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
    if len(paket_final) < jumlah_tampil:
        for paket in kolam_kandidat:
            kombinasi_id = f"{paket['Main Course']['nama_menu']}-{paket['Beverage']['nama_menu']}-{paket['Side Dish']['nama_menu']}"
            terdaftar = [f"{p['Main Course']['nama_menu']}-{p['Beverage']['nama_menu']}-{p['Side Dish']['nama_menu']}" for p in paket_final]
            
            if kombinasi_id not in terdaftar:
                paket_final.append(paket)
                
            if len(paket_final) == jumlah_tampil:
                break
                
    return paket_final
