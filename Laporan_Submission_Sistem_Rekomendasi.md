
# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

## Project Overview

Sistem rekomendasi merupakan salah satu aplikasi populer dalam machine learning, khususnya dalam domain e-commerce, hiburan, dan media digital. Dalam proyek ini, dikembangkan sistem rekomendasi film berdasarkan interaksi pengguna dengan berbagai judul film.

Permasalahan muncul ketika pengguna kesulitan menemukan film yang relevan dengan preferensi mereka, terutama dalam jumlah koleksi film yang sangat besar. Untuk itu, sistem rekomendasi berbasis data rating pengguna dapat membantu menyaring dan merekomendasikan film yang sesuai.

Sumber data yang digunakan berasal dari dataset **MovieLens 100K**, yang secara luas digunakan untuk pengembangan dan evaluasi sistem rekomendasi.

## Business Understanding

### Problem Statements

- Bagaimana cara merekomendasikan film yang relevan kepada pengguna berdasarkan preferensi mereka?
- Metode rekomendasi apa yang paling efektif untuk dataset ini?

### Goals

- Mengembangkan sistem rekomendasi yang mampu menyarankan film dengan akurasi tinggi.
- Membandingkan performa beberapa algoritma sistem rekomendasi.

### Solution Approach

#### Solution Statements

- Collaborative Filtering berbasis **user-item interactions** (K-Nearest Neighbors).
- Matrix Factorization dengan algoritma **Singular Value Decomposition (SVD)** dari pustaka `Surprise`.

## Data Understanding

Dataset yang digunakan adalah **MovieLens 100K**, yang terdiri dari 100.000 data rating dari 943 pengguna terhadap 1682 film.

Link dataset: [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)

Beberapa fitur utama dalam dataset ini adalah:

- `user_id`: ID unik pengguna
- `item_id`: ID unik film
- `rating`: Skor yang diberikan pengguna terhadap film (1–5)
- `timestamp`: Waktu rating diberikan

Visualisasi dan analisis eksploratori data menunjukkan bahwa:

- Distribusi rating cenderung normal dengan puncak di skor 4.
- Tidak semua pengguna menilai semua film, sehingga data bersifat sparse.

## Data Preparation

Beberapa langkah yang dilakukan dalam data preparation meliputi:

- **Membaca dan memformat data**: Menggunakan `pandas` untuk membaca file dan menyesuaikan kolom sesuai format pustaka `Surprise`.
- **Mengekstrak ID pengguna dan film**: Disesuaikan agar bisa dibaca oleh model.
- **Membagi data**: Menggunakan 80% data untuk pelatihan dan 20% untuk pengujian menggunakan `train_test_split`.

Langkah-langkah ini diperlukan agar data kompatibel dengan model sistem rekomendasi yang digunakan dan untuk menjamin evaluasi yang adil.

## Modeling

Model yang digunakan dalam proyek ini:

1. **K-Nearest Neighbors (KNN)** dengan pendekatan user-based collaborative filtering.
2. **Singular Value Decomposition (SVD)** untuk matrix factorization.

Output dari model berupa rekomendasi top-N film untuk pengguna berdasarkan rating yang diprediksi.

**Contoh output:**
```
Rekomendasi untuk user ID 196:
1. Star Wars (1977)
2. Return of the Jedi (1983)
3. Raiders of the Lost Ark (1981)
```

Kelebihan & Kekurangan:

| Metode | Kelebihan | Kekurangan |
|--------|-----------|------------|
| KNN    | Interpretatif, sederhana | Kurang efektif untuk data besar |
| SVD    | Akurasi tinggi, cocok untuk data sparse | Perlu waktu pelatihan lebih lama |

## Evaluation

Metrik evaluasi yang digunakan adalah **Root Mean Square Error (RMSE)** dan **Mean Absolute Error (MAE)**:

- RMSE menunjukkan seberapa jauh prediksi dari nilai sebenarnya, dengan penalti lebih besar untuk error besar.
- MAE menghitung rata-rata selisih absolut antara prediksi dan nilai aktual.

**Hasil Evaluasi:**

| Model | RMSE | MAE |
|-------|------|-----|
| KNN   | 1.01 | 0.80 |
| SVD   | 0.93 | 0.74 |
