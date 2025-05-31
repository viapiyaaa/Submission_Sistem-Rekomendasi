# Laporan Proyek Machine Learning - Evi Afiyatus Solihah

## Project Overview

Sistem rekomendasi merupakan salah satu aplikasi populer dalam machine learning, khususnya di bidang e-commerce, hiburan, dan media digital. Dalam proyek ini dikembangkan sistem rekomendasi film yang didasarkan pada interaksi pengguna terhadap berbagai judul film. Banyaknya pilihan film seringkali membuat pengguna kesulitan menemukan tontonan yang sesuai, sehingga dapat mengakibatkan pemborosan waktu dan ketidakpuasan (Fajriansyah, Adikara, & Widodo, 2021). Untuk mengatasi masalah tersebut, proyek ini membangun dua sistem rekomendasi terpisah: Content-Based Filtering yang memanfaatkan kemiripan genre film melalui TF-IDF yang memberikan bobot pada fitur-fitur dan cosine similarity digunakan untuk mengukur kesamaan antara film yang dipilih pengguna dengan film lainnya berdasarkan dataset (Sari, Isnaini, & Seniwati, 2025), serta Collaborative Filtering berbasis neural network menggunakan teknik embedding untuk merepresentasikan pengguna dan item dalam bentuk vektor. Pengguna dengan tingkat kesamaan tertinggi berdasarkan representasi embedding tersebut dikelompokkan, lalu direkomendasikan item yang disukai oleh pengguna lain dalam kelompok tersebut berdasarkan data rating yang mereka berikan (Nugroho, Lubis, & Perdana, 2024).  

## Business Understanding

Pengguna sering kali mengalami kesulitan dalam menemukan film yang sesuai dengan preferensinya di tengah jumlah film yang sangat banyak. Hal ini dapat menyebabkan ketidakpuasan pengguna, rendahnya engagement, serta potensi kehilangan peluang untuk konsumsi konten lebih lanjut.

### Problem Statements

Mengacu pada latar belakang yang telah dijelaskan, berikut adalah sejumlah permasalahan yang dapat diselesaikan melalui proyek ini.

- Bagaimana sistem dapat merekomendasikan film yang relevan dengan preferensi pengguna berdasarkan kesamaan konten, seperti genre, meskipun pengguna memiliki sedikit atau tanpa riwayat rating sebelumnya?
- Bagaimana sistem dapat mempelajari pola perilaku rating dari banyak pengguna untuk memprediksi film yang kemungkinan besar akan disukai oleh pengguna tertentu?

### Goals

Adapun tujuan dari proyek ini adalah sebagai berikut.

- Mengembangkan sistem rekomendasi berbasis konten (Content-Based Filtering) yang mampu merekomendasikan film kepada pengguna berdasarkan kemiripan atribut film,
  seperti genre, tanpa bergantung pada riwayat interaksi yang banyak.
- Membangun model rekomendasi berbasis kolaboratif (Collaborative Filtering) yang dapat mempelajari pola rating dari banyak pengguna untuk memberikan rekomendasi film
  yang dipersonalisasi.

    ### Solution statements
    - Untuk memahami karakteristik data dalam sistem rekomendasi, dilakukan eksplorasi dan visualisasi data terkait pola interaksi antara pengguna dan film. Analisis ini mencakup distribusi rating, frekuensi pemberian rating oleh pengguna, serta popularitas film berdasarkan jumlah rating yang diterima.
    - Melakukan pembersihan data dengan mengecek dan menghapus nilai kosong serta data duplikat untuk memastikan kualitas data
    - Dalam membangun sistem rekomendasi film, digunakan dua pendekatan utama untuk memprediksi preferensi pengguna, yaitu Content-Based Filtering dan Collaborative Filtering:
       - Content-Based Filtering: menggunakan TF-IDF vectorization terhadap genre film dan cosine similarity untuk menghitung kemiripan antar film.
       - Collaborative Filtering: membangun model RecommenderNet menggunakan TensorFlow dan layer embedding untuk mempelajari interaksi user-item.

## Data Understanding

Dataset yang digunakan berisi data film, rating, tag, dan link yang diunduh dari Kaggle [Movie Lens Dataset](https://www.kaggle.com/datasets/aigamer/movie-lens-dataset). Dataset ini terdiri dari: 

- movies.csv: merupakan informasi metadata film seperti movieId, title, dan genres.
- ratings.csv: merupakan data interaksi pengguna dan film berupa userId, movieId, rating, dan timestamp.
- tag.csv: merupakan tag atau label yang diberikan pengguna pada film. userId, movieId, tag, dan timestamp.
- link.csv: merupakan mapping movieId ke IMDb dan TMDb untuk integrasi data eksternal. movieId, imdbId, dan tmdbId.
  
Adapun beberapa fitur yang digunakan dalam pembuatan sistem rekomendasi ini adalah sebagai berikut:

- movieId: ID unik setiap film.
- title: Judul film.
- genres: Genre film (bisa lebih dari satu).
- userId: ID unik pengguna.
- rating: Penilaian yang diberikan pengguna terhadap film (skala 0.5 - 5.0).

Pada tahap ini dilakukan eksplorasi data untuk memahami isi dan karakteristik dataset. Pertama, informasi dari setiap file .csv diperiksa menggunakan fungsi seperti .info() dan .describe(). Selanjutnya, data digabungkan dan diperiksa apakah ada data duplikat atau nilai yang kosong menggunakan fungsi isnull(). Berikut hasil dataset setelah penggabungan.

![image](https://github.com/user-attachments/assets/bd172f04-3bb4-4889-9c7f-faf6bed04e9d)

Gambar 1. Dataset setelah digabungkan (movie + ratings + tag)

Kemudian, dilakukan analisis pada data rating, seperti menghitung jumlah rating yang diberikan oleh setiap pengguna dan setiap film, serta rata-rata rating. Distribusi skor rating juga divisualisasikan untuk memudahkan pemahaman. Selain itu, ditampilkan 20 film dengan jumlah rating terbanyak dan 20 pengguna yang paling aktif memberikan rating, lengkap dengan visualisasinya. Adapun hasil dari eksplorasi tersebut adalah sebagai berikut.

![image](https://github.com/user-attachments/assets/7fa66560-abbd-46c8-b37c-27e89984d647)

Gambar 2a. Sebaran skor rating

Secara keseluruhan, gambar grafik diatas menggambarkan bahwa penonton cenderung memberikan rating yang positif atau cukup positif terhadap film. Mayoritas film mendapatkan rating 3, 4, atau 5 bintang, dengan 4 bintang menjadi rating yang paling sering diberikan. Ini bisa berarti bahwa rata-rata film yang tersedia memiliki kualitas yang dianggap "baik" oleh sebagian besar penonton, atau bahwa penonton cenderung memberikan rating yang lebih tinggi.

![image](https://github.com/user-attachments/assets/d8d8fd26-ecd5-4882-b5ea-08fb65848a11)

Gambar 2b. 20 Film dengan rating terbanyak

Secara keseluruhan, grafik ini menggambarkan film-film yang paling sering diulas atau diberi rating oleh pengguna, menyoroti popularitas dan daya tarik abadi dari sejumlah film klasik, terutama dari era 1990-an. Data ini menunjukkan bahwa film-film klasik dari era tersebut sangat populer di kalangan pemberi rating, dan penonton secara umum cenderung memberikan ulasan yang positif atau sangat baik terhadap film-film yang mereka tonton. Dalam dataset ini, film-film populer yang paling banyak mendapatkan rating, seperti "Forrest Gump," "Shawshank Redemption," dan "Pulp Fiction," didominasi oleh rilis tahun 1990-an, terutama dari tahun 1994, yang menunjukkan daya tarik abadi dari karya-karya klasik tersebut. Meskipun mencakup berbagai genre, daftar film dengan rating terbanyak ini secara tidak langsung merefleksikan popularitas tinggi di kalangan penonton aktif yang memberikan rating, dengan rentang jumlah rating antara 200 hingga lebih dari 300.

![image](https://github.com/user-attachments/assets/2e9153d1-242f-462b-a620-d98a45ef5bb8)

Gambar 2c. 20 Pengguna aktif berdasarkan rating terbanyak

Secara keseluruhan, grafik ini mengungkapkan bahwa platform rating film ini memiliki sejumlah kecil pengguna yang sangat aktif dan berdedikasi yang berkontribusi secara signifikan terhadap volume data rating. Kondisi ini berimplikasi pada potensi bias dalam statistik keseluruhan, karena preferensi atau karakteristik rating dari kelompok inti pengguna ini dapat secara signifikan memengaruhi tren yang diamati. Oleh karena itu, saat menganalisis data dari platform ini, sangat krusial untuk memahami bahwa meskipun volume rating tinggi, distribusinya tidak merata dan sangat bergantung pada kontribusi berharga dari segelintir pengguna inti tersebut.

Analisis juga dilakukan pada genre film dengan melihat 20 genre yang paling sering muncul. Berikut merupakan visualisasi hasil analisis tersebut.

![image](https://github.com/user-attachments/assets/eb962aea-137d-444b-b044-663e72254cd0)

Gambar 3. Sebaran 20 Genre film terbanyak 

Secara keseluruhan, grafik ini dengan jelas menunjukkan bahwa genre Drama dan Comedy adalah pilar utama dalam kumpulan data film ini, diikuti oleh genre populer lainnya seperti Thriller dan Action, sementara banyak genre lain memiliki representasi yang jauh lebih kecil. Dalam dataset film ini, genre Drama dan Comedy mendominasi dengan jumlah film masing-masing melebihi dan mendekati 4000, menunjukkan produktivitas atau popularitas yang signifikan. Genre seperti Thriller dan Action juga memiliki representasi kuat dengan sekitar 1800-2000 film. Namun, terdapat rentang popularitas yang luas, di mana genre minoritas seperti Film-Noir, IMAX, Western, dan Musical memiliki jumlah film yang jauh lebih sedikit, hanya beberapa ratus atau kurang. Perbedaan ini menyoroti fokus industri pada genre tertentu, yang penting untuk dipertimbangkan dalam sistem rekomendasi guna memastikan variasi dan relevansi genre minoritas bagi pengguna.

Pada tahap ini juga ditentukan fitur-fitur akhir yang akan digunakan dalam pembuatan sistem rekomendasi, yaitu movieId, title, genres, userId, dan rating.

## Data Preparation

Pada tahap data preparation dilakukan penghapusan data duplikat dan nilai kosong (null), proses encoding terhadap userId dan movieId, pembuatan matriks TF-IDF dari kolom genres, serta pembagian data menjadi data pelatihan dan validasi. Setelah itu, dilakukan perhitungan cosine similarity berdasarkan matriks TF-IDF untuk mengukur tingkat kemiripan antar film berdasarkan genre-nya.

Tidak terdapat nilai null pada fitur-fitur yang digunakna, tetpi terdapat data duplikat sebanyak 1841 yang berarti ada sejumlah interaksi pengguna dan film yang tercatat lebih dari satu kali dengan informasi yang sama persis (userId, movieId, title, genres, dan rating). Jumlah data setelah penghapusan duplikat adalah 100.836. 

Langkah persiapan data untuk sistem rekomendasi Content-Based Filtering meliputi pembuatan matriks TF-IDF dari kolom genres pada dataset film. Matriks ini merepresentasikan setiap film dalam bentuk vektor berdasarkan genre-nya. Selanjutnya, dilakukan perhitungan cosine similarity antar film berdasarkan vektor TF-IDF tersebut untuk mengukur tingkat kemiripan konten antar film. Nilai similarity ini kemudian digunakan untuk merekomendasikan film yang memiliki genre serupa dengan film yang pernah disukai atau ditonton oleh pengguna.

![image](https://github.com/user-attachments/assets/07b3432e-ca23-48fe-a80d-2d089c05f9d9)

Gambar 4. Gambar hasil representasi vektor fitur genre film dengan TF-IDF

![image](https://github.com/user-attachments/assets/78db81da-e984-404b-8c55-b1d1ce23e1b7)

Gambar 5. Hasil Similarity matrix pada setiap film

Adapun proses encoding terhadap userId dan movieId dilakukan untuk mengubahnya menjadi ID numerik yang berurutan. Tujuannya adalah agar data tersebut dapat digunakan dalam layer embedding pada model sistem rekomendasi dengan pendekatan collaborative filtering. Setelah itu, data dibagi menjadi data pelatihan dan data validasi untuk sebesar 80:20 keperluan pelatihan model.

## Modeling

Dalam membangun sistem rekomendasi film, digunakan dua pendekatan utama untuk memprediksi preferensi pengguna, yaitu Content-Based Filtering dan Collaborative Filtering, dengan penjelasan sebagai berikut:

**Content-Based Filtering:**

Pada pendekatan ini dilakukan pembuatan matriks TF-IDF dari kolom genres pada dataset film. Matriks ini merepresentasikan setiap film dalam bentuk vektor berdasarkan genre-nya. Selanjutnya, dilakukan perhitungan cosine similarity antar film berdasarkan vektor TF-IDF tersebut untuk mengukur tingkat kemiripan konten antar film. Nilai similarity ini kemudian digunakan untuk merekomendasikan film yang memiliki genre serupa dengan film yang pernah disukai atau ditonton oleh pengguna.

Fungsi movie_recommendations() digunakan dalam content-based filtering untuk merekomendasikan film yang mirip berdasarkan genre. Dengan input berupa judul film dan matriks kemiripan, fungsi ini mencari film-film dengan skor kemiripan tertinggi, lalu mengembalikan daftar rekomendasi yang telah digabung dengan metadata seperti genre.

Kelebihan:
- Bisa merekomendasikan film baru atau untuk pengguna baru selama data fitur (misal genre, aktor, deskripsi) tersedia.
- Rekomendasi berdasarkan kesamaan fitur film, jadi jelas alasan rekomendasinya.
- Tidak bergantung pada data interaksi pengguna lain, sehingga tidak terpengaruh sparsity data.

Kekurangan:
- Terbatas pada rekomendasi yang mirip dengan film yang sudah pernah ditonton, jadi kurang eksplorasi.
- Kualitas rekomendasi sangat bergantung pada kualitas dan kelengkapan data fitur film.
- Sulit menangkap preferensi kompleks yang tidak hanya berdasarkan konten, misalnya mood atau konteks pengguna.

`movie_recommendations('Mad Max (1979)')`

![image](https://github.com/user-attachments/assets/29998bce-aab0-47af-9a0c-0f26af20d24b)

Gambar 6. Hasil rekomendasi film berdasarkan pendekatan Content-Based Filtering

**Collaborative Filtering:**

Menggunakan neural network sederhana dengan embedding layer (RecommenderNet) untuk mempelajari hubungan antar pengguna dan film. Model menerima pasangan [user, movie] yang sudah di-encoding menjadi format numerik, lalu memprediksi rating. Dataset interaksi pengguna dengan film dibagi menjadi data latih dan data uji untuk melatih dan menguji model. Model belajar dari pola rating atau interaksi pengguna untuk memprediksi film yang mungkin disukai. Output akhir berupa rekomendasi top-N film berdasarkan prediksi rating tertinggi yang disesuaikan dengan preferensi pengguna dan pola interaksi pengguna lain yang serupa.

Kelebihan:
- Tidak membutuhkan data fitur film (konten) secara eksplisit, cukup berdasarkan interaksi pengguna (rating, klik, dsb).
- Bisa menemukan rekomendasi yang tidak hanya mirip konten, tapi juga berdasarkan preferensi pengguna lain yang mirip, jadi lebih personal.
- Mampu merekomendasikan film dengan genre atau tipe yang berbeda dari yang pernah ditonton user, sehingga bisa eksplorasi lebih luas.

Kekurangan:
- Mengalami masalah cold start untuk pengguna baru yang belum punya riwayat interaksi atau film baru yang belum banyak di-rating.
- Butuh data interaksi yang cukup banyak agar model bisa belajar dengan baik.
- Rentan terhadap data sparsity (data interaksi yang sangat jarang), sehingga kualitas rekomendasi bisa menurun.

![image](https://github.com/user-attachments/assets/4c678822-ebf9-4337-b7ef-bde69a1065b0)

Gambar 7. Hasil rekomendasi film berdasarkan pendekatan Collaborative Filtering

## Evaluation

Model Collaborative Filtering dievaluasi menggunakan metrik Root Mean Squared Error (RMSE), yang mengukur seberapa jauh prediksi rating dari nilai rating sebenarnya. RMSE yang lebih rendah menunjukkan kualitas prediksi yang lebih baik.

Hasil pelatihan menunjukkan bahwa model mencapai performa yang cukup baik:

RMSE Training (akhir epoch): 0.1372
RMSE Validation (akhir epoch): 0.1923

Meskipun terdapat sedikit perbedaan antara error pada data pelatihan dan data validasi, selisih tersebut masih dalam batas wajar dan tidak menunjukkan overfitting yang signifikan. Hal ini mengindikasikan bahwa model memiliki kemampuan generalisasi yang cukup baik terhadap data baru. Namun, untuk meningkatkan performa lebih lanjut dan mencegah potensi overfitting pada pelatihan jangka panjang, teknik tambahan seperti Dropout, regularisasi (L2), atau early stopping dapat diterapkan. Selain itu, peningkatan kualitas data dan penyempurnaan dimensi embedding juga dapat menjadi strategi pengembangan model selanjutnya.

![image](https://github.com/user-attachments/assets/12de2cfa-40f2-4e3f-b89d-e7021424bd83)

Gambar 8. Grafik Hasil Pelatihan 

Grafik diatas menunjukkan bagaimana RMSE berubah seiring bertambahnya epoch dalam pelatihan model. Garis merah merepresentasikan RMSE training, yang menurun secara konsisten, menandakan bahwa model semakin memahami pola dalam data latihannya. Sementara itu, garis biru menunjukkan RMSE validasi, yang juga menurun tetapi dengan laju lebih lambat. Hal ini berarti model mulai bekerja lebih baik dengan data yang belum pernah dilihat sebelumnya. Secara keseluruhan, grafik ini menunjukkan progres yang baik, tetapi penting untuk terus memantau agar keseimbangan antara training dan validasi tetap optimal.

## **Referensi**
Fajriansyah, M., Adikara, P. P., & Widodo, A. W. (2021). Sistem Rekomendasi Film Menggunakan Content Based Filtering. Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer, 5(6), 2188-2199.
Sari, R. A., Isnaini, S. F., & Seniwati, E. (2025). Sistem Rekomendasi Film Menggunakan Content Based Filtering. The Indonesian Journal of Computer Science Research, 4(1), 19-27.
Nugroho, D. A., Lubis, C., & Perdana, N. J. (2024). Sistem Rekomendasi Film Menggunakan Metode Neural Collaborative Filtering Movie Recommendation System Using Neural Collaborative Filtering. J. Inf. Technol. Comput. Sci, 7(3), 6765-6775.

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.


