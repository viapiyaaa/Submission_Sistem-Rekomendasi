# Laporan Proyek Machine Learning - Evi Afiyatus Solihah

## Domain Proyek

Domain yang dipilih untuk proyek machine learning ini adalah Kesehatan, dengan judul Predictive Analytics: Klasifikasi Level Obesitas (Underweight, Normal, Overweight, Obese)

# Latar Belakang

Obesitas telah mengalami peningkatan yang signifikan dalam sepuluh tahun terakhir dan menjadi salah satu isu utama dalam bidang kesehatan masyarakat, baik secara global maupun di Indonesia (Aini, Khasanah, Ristyawan, & Diniati, 2024). Kondisi ini berkontribusi terhadap berbagai penyakit kronis, seperti diabetes, penyakit jantung, dan hipertensi. Obesitas merupakan kondisi kelebihan berat badan yang berdampak negatif terhadap kesehatan, dan dipengaruhi oleh berbagai faktor, termasuk kondisi fisik seperti tinggi badan, berat badan, dan Indeks Massa Tubuh (IMT), serta faktor lainnya seperti jenis kelamin dan tingkat aktivitas fisik  (Setiyani, Indahsari, & Roestam, 2023). Deteksi dini terhadap kategori obesitas sangat penting untuk mendukung upaya intervensi dan pencegahan yang efektif. Dengan memanfaatkan data demografis dan karakteristik fisik, proyek ini bertujuan untuk membangun model klasifikasi yang mampu memprediksi kategori obesitas—Underweight, Normal weight, Overweight, dan Obese—secara akurat. Penelitian sebelumnya menunjukkan bahwa pendekatan machine learning dapat digunakan secara efektif untuk mengklasifikasikan status obesitas berdasarkan data karakteristik individu (Sitanggang & Sherly, 2022). Dalam proyek ini, analisis dilakukan berdasarkan dataset yang diperoleh dari platform Kaggle.

## Business Understanding

Pengembangan model prediksi tingkat obesitas memiliki potensi besar untuk membantu berbagai pihak, seperti tenaga medis, lembaga kesehatan, dan individu. Model ini dapat digunakan untuk mendeteksi status obesitas secara cepat dan akurat, sehingga mendukung intervensi dini, pencegahan penyakit kronis, dan peningkatan kesadaran akan pola hidup sehat. Hasil prediksi yang tepat juga dapat membantu dalam pemantauan status kesehatan masyarakat, penyusunan kebijakan kesehatan, serta pengambilan keputusan klinis yang lebih efisien.

### Problem Statements
Mengacu pada latar belakang yang telah dijelaskan, berikut adalah sejumlah permasalahan yang dapat diselesaikan melalui proyek ini.
- Bagaimana mengklasifikasikan status obesitas individu berdasarkan data fisik dan aktivitas?
- Bagaimana meningkatkan akurasi prediksi kategori obesitas dengan metode machine learning?

### Goals
Adapun tujuan dari proyek ini adalah sebagai berikut.
- Membangun model klasifikasi untuk mengkategorikan status obesitas dengan akurasi tinggi.
- Mengidentifikasi fitur yang paling berpengaruh dalam prediksi obesitas.

  ### Solution statements
- Untuk menjawab permasalahan yang ada, dilakukan analisis univariat dan multivariat guna memahami karakteristik data. Visualisasi digunakan sebagai alat bantu dalam mengevaluasi korelasi antar fitur dan mengidentifikasi data pencilan (outlier) yang berpotensi mengganggu akurasi model.
- Dilakukan proses data cleaning dan standardisasi data untuk memastikan kualitas data yang baik.
- Beragam model dibangun dan dievaluasi guna memperoleh model paling optimal untuk prediksi kualitas apel. Di antaranya adalah:
    - Random Forest adalah algoritma berbasis pohon keputusan yang menggabungkan banyak pohon untuk menghasilkan prediksi yang lebih akurat. Cocok untuk klasifikasi obesitas karena dapat menangani banyak fitur dan hubungan non-linear antar variabel.
    - SVM bekerja dengan mencari garis pemisah terbaik antar kelas. Dalam klasifikasi obesitas, SVM efektif memisahkan level obesitas berdasarkan fitur seperti BMI, usia, dan aktivitas fisik, terutama jika data memiliki batas kelas yang jelas.
    - KNN mengklasifikasikan data berdasarkan kedekatan dengan tetangga terdekat. Untuk prediksi obesitas, KNN menilai individu berdasarkan kemiripannya dengan data yang telah diberi label sebelumnya.

## Data Understanding

Dataset yang digunakan berisi data fisik dan demografi dari individu yang telah diberi label kategori obesitas. Dataset ini dapat diunduh dari Kaggle [obesity-prediction](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction/).

# EDA - Deskripsi Variabel

**Tabel Informasi Dataset**

| **Informasi**       | **Deskripsi**                                                           |
|---------------------|-------------------------------------------------------------------------|
| **Title**           | Obesity Prediction                                                      |
| **Source**          | Kaggle                                                                  |
| **Maintainer**      | MrSimple07                                                              |
| **License**         | Other (specified in description)                                        |
| **Visibility**      | Publik                                                                  |
| **Tags**            | Health, Health Conditions                                               |
| **Usability Score** | 10.00 (semakin tinggi semakin mudah digunakan, skala maksimal adalah 10)| 

Adapun isi dari dataset yang digunakan dalam proyek ini adalah Obesity Prediction yang bersumber dari platform Kaggle dan dikelola oleh pengguna dengan nama MrSimple07.

**Tabel Isi Dataset**

| Age | Gender | Height     | Weight     | BMI        | PhysicalActivityLevel | ObesityCategory |
|-----|--------|------------|------------|------------|----------------------|-----------------|
| 56  | Male   | 173.575262 | 71.982051  | 23.891783  | 4                    | Normal weight   |
| 69  | Male   | 164.127306 | 89.959256  | 33.395209  | 2                    | Obese           |
| 46  | Female | 168.072202 | 72.930629  | 25.817737  | 4                    | Overweight      |
| 32  | Male   | 168.459633 | 84.886912  | 29.912247  | 3                    | Overweight      |
| 60  | Male   | 183.568568 | 69.038945  | 20.487903  | 3                    | Normal weight   |
| 25  | Female | 166.405627 | 61.145868  | 22.081628  | 4                    | Normal weight   |
| 78  | Male   | 183.566334 | 92.208521  | 27.364341  | 3                    | Overweight      |
| 38  | Male   | 142.875095 | 59.359746  | 29.078966  | 1                    | Overweight      |
| 56  | Male   | 183.478558 | 75.157672  | 22.325577  | 4                    | Normal weight   |
| 75  | Male   | 182.974061 | 81.533460  | 24.353244  | 2                    | Normal weight   |

Dataset ini memiliki total 1000 entri (baris) dan mencakup 7 fitur (kolom), dengan deskripsi masing-masing kolom dijelaskan di bawah ini.

- **Age** : Usia individu (tahun)
- **Gender** : Jenis kelamin (Laki-laki/Perempuan)
- **Height** : Tinggi badan (cm)
- **Weight** : Berat badan (kg)
- **BMI** : Indeks Massa Tubuh
- **PhysicalActivityLevel** : Tingkat aktivitas fisik harian
- **ObesityCategory** : Kategori obesitas (Underweight, Normal weight, Overweight, Obese)

Selain itu, dilakukan exploratory data analysis untuk memeriksa distribusi data, nilai hilang, outlier, dan korelasi antar fitur. Juga dilakukan visualisasi data.

# EDA - Exploratory Data Analysis Menangani Missing Value

Dalam tahap Exploratory Data Analysis (EDA), dilakukan beberapa langkah utama, yaitu Data Gathering, Data Assessing, dan Data Cleaning. 
- Data Gathering: Pada tahap ini, data diimpor dan disusun agar dapat dibaca dengan baik menggunakan objek DataFrame dari library Pandas.
- Data Assessing: Dilakukan evaluasi terhadap kualitas dan integritas data, termasuk pengecekan terhadap:
  - Duplikasi data, yaitu entri yang sama muncul lebih dari sekali,
  - Missing value, yaitu nilai yang tidak tersedia dalam dataset, dan
  - Outlier, yaitu data yang menyimpang jauh dari pola umum.
- Data Cleaning: Tahap ini mencakup penanganan terhadap data duplikat, nilai hilang (missing values), dan outlier untuk memastikan data bersih dan siap digunakan dalam proses analisis dan pemodelan lebih lanjut.

Pada proyek kasus ini, tidak ditemukan adanya data duplikat maupun nilai yang hilang (missing value). Namun, terdapat outlier yang akan diatasi menggunakan metode dropping berdasarkan metode Interquartile Range (IQR). 

Pada tahap ini juga dilihat jumlah masing- masing dari tingkat obesitas yaitu Normal weight, Overweight, Obese, dan Underweight, yang ditunjukkan pada tabel di bawwah ini.

| ObesityCategory        | Count |
|------------------------|-------|
| Normal weight          | 371   |
| Overweight             | 293   |
| Obese                  | 175   |
| Underweight            | 135   |

# EDA - Univariate Analysis

![Gambar 1a. Analisis Univariat (Data Kategori: Gender)](https://github.com/user-attachments/assets/c067f9b3-84b3-472c-912b-6b60c2238a2b)

**Gambar 1a. Analisis Univariat (Data Kategori: Gender**

Terlihat bahwa jumlah laki-laki (Male) sedikit lebih banyak dibandingkan perempuan (Female), dengan sekitar 510 laki-laki dan 460 perempuan. Perbedaan ini tidak terlalu signifikan, sehingga distribusi gender dalam data cukup seimbang.

![Gambar 1b. Analisis Univariat (Data Kategori: ObesityCategory)](https://github.com/user-attachments/assets/b71c3b0a-89d4-4439-b6d8-200dbe81c713)

**Gambar 1b. Analisis Univariat (Data Kategori: ObesityCategory)**

Grafik di atas menunjukkan distribusi kategori obesitas dalam dataset. Kategori dengan jumlah tertinggi adalah Normal weight, diikuti oleh Overweight, Obese, dan yang paling sedikit adalah Underweight. Hal ini menunjukkan bahwa mayoritas individu dalam dataset memiliki berat badan normal, sedangkan jumlah penderita obesitas dan underweight relatif lebih sedikit. 

![Gambar 2a. Analisis  Univariat (Data Numerik: Age dan Height)](https://github.com/user-attachments/assets/49177494-3ca2-489b-8399-4336b4761b02)

**Gambar 2a. Analisis  Univariat (Data Numerik: Age dan Height)**

Pada grafik sebelah kiri, distribusi usia tampak cukup merata dengan sedikit puncak di usia tertentu seperti sekitar 20, 30, 40, dan 70 tahun. Hal ini menunjukkan bahwa tidak ada dominasi usia tertentu, namun ada kecenderungan kelompok usia tertentu muncul lebih sering, mungkin karena faktor sampling atau karakteristik populasi yang dikaji.

Sementara itu, grafik sebelah kanan menunjukkan distribusi tinggi badan yang menyerupai bentuk distribusi normal (bell curve), dengan puncaknya berada di sekitar 165–170 cm. Ini menandakan bahwa sebagian besar individu memiliki tinggi badan di kisaran tersebut, dan jumlah individu berkurang seiring dengan tinggi yang sangat pendek maupun sangat tinggi.

![Gambar 2b. Analisis  Univariat (Data Numerik: Weight dan BMI)](https://github.com/user-attachments/assets/bd8eddba-9e31-40af-a467-c1c95c596586)

**Gambar 2b. Analisis  Univariat (Data Numerik: Weight dan BMI)**

Histogram kiri berjudul "Weight" yang menampilkan distribusi berat badan dengan rentang sekitar 30 sampai 110 (mungkin dalam satuan kg). Distribusinya terlihat miring sedikit ke kiri dengan puncak di sekitar 75-80.

Histogram kanan berjudul "BMI" yang menunjukkan distribusi Indeks Massa Tubuh (BMI) dengan rentang dari sekitar 10 sampai 40. Distribusinya juga menyerupai distribusi normal dengan puncak sekitar 24-26.

![Gambar 2c. Analisis  Univariat (Data Numerik: PhysicalActivityLevel dan ObesityCategory_num)](https://github.com/user-attachments/assets/de5d7457-4eaf-4103-9e8d-dad5309ccaef)

**Gambar 2c. Analisis  Univariat (Data Numerik: PhysicalActivityLevel dan ObesityCategory_num)**


# EDA - Multivariat Analysis
![Gambar 3a Jumlah Obesity Category berdasarkan Gender](https://github.com/user-attachments/assets/7d9925ae-20ca-4f7d-a9bf-3b368e9b16a4)

**Gambar 3a Jumlah Obesity Category berdasarkan Gender**

Grafik di atas menunjukkan distribusi kategori obesitas berdasarkan gender. Terlihat bahwa baik pada pria maupun wanita, kategori dengan jumlah individu terbanyak adalah Normal weight. Pada pria, jumlah individu dengan berat badan normal lebih tinggi dibanding wanita, dengan selisih yang cukup signifikan. Sementara itu, kategori Overweight lebih banyak ditemukan pada wanita dibandingkan pria, menunjukkan kecenderungan kelebihan berat badan yang sedikit lebih besar di kelompok wanita. Untuk kategori Obese dan Underweight, jumlahnya relatif seimbang antara pria dan wanita, meskipun tetap menunjukkan sedikit penurunan pada kelompok wanita.

![image](https://github.com/user-attachments/assets/9bc45493-9a38-447b-a87b-a20e836531db)

![Gambar 3b hubungan antar fitur numerik](https://github.com/user-attachments/assets/09bd8691-a53c-4b22-8088-4d043a9a222a)

**Gambar 3b hubungan antar fitur numerik**

Gambar pairplot menunjukkan hubungan antar fitur numerik dalam dataset. Terlihat korelasi positif yang kuat antara Weight dan BMI, serta antara BMI dan ObesityCategory_num, yang wajar karena kategori obesitas diturunkan dari nilai BMI. Sebaliknya, Height berkorelasi negatif dengan BMI. Distribusi tiap fitur juga tampak normal, terutama pada Height, Weight, dan BMI. Sementara PhysicalActivityLevel tidak menunjukkan pola korelasi yang jelas dengan fitur lain, menunjukkan pengaruhnya terhadap obesitas mungkin tidak langsung. Visualisasi ini membantu memahami data dan memilih fitur yang relevan untuk pemodelan.

![Gambar 3c Correlation Matrix untuk Fitur Numerik](https://github.com/user-attachments/assets/5742a8b4-ed08-4d76-ae90-3721d6a3f367)

**Gambar 3c Correlation Matrix untuk Fitur Numerik**
 
Berdasarkan hasil visualisasi matriks korelasi pada fitur numerik, dapat disimpulkan bahwa fitur yang paling berpengaruh dalam menentukan tingkat obesitas (ObesityCategory_num) adalah BMI, dengan nilai korelasi sebesar 0,94. Hal ini menunjukkan bahwa BMI memiliki hubungan yang sangat kuat dengan klasifikasi tingkat obesitas. Selain itu, berat badan (Weight) juga menunjukkan korelasi yang cukup tinggi terhadap tingkat obesitas, yaitu sebesar 0,82, sedangkan tinggi badan (Height) memiliki korelasi negatif sebesar -0,40. Korelasi negatif ini wajar, mengingat tinggi badan merupakan salah satu komponen dalam perhitungan BMI.

Sebaliknya, fitur usia (Age) dan tingkat aktivitas fisik (PhysicalActivityLevel) menunjukkan korelasi yang sangat rendah terhadap tingkat obesitas, masing-masing sebesar -0,06 dan 0,03. Ini mengindikasikan bahwa pengaruh keduanya terhadap klasifikasi obesitas dalam dataset ini relatif kecil. Dengan demikian, fitur BMI, berat badan, dan tinggi badan dapat dianggap sebagai variabel paling penting, dan sebaiknya dijadikan fokus utama dalam pembangunan model prediktif tingkat obesitas.

## Data Preparation

Pada tahap preparation dilakukan penghapusan outlier dengan metode Interquartile Range (IQR), menggabungkan fitur 'Weight' dan 'BMI' menjadi fitur baru 'WeightBMICombined' dengan melakukan Principal Component Analysis (PCA), melakukan pemetaan kategori ObesityCategory ke angka di kolom ObesityCategory_num, dan transformasi data seperti One-Hot Encoding pada variabel kategorikal, , agar variabel tersebut dapat direpresentasikan dalam bentuk numerik yang sesuai untuk pelatihan model machine learning.

Untuk menghapus outlier digunakan metode Interquartile Range (IQR). Metode ini dilakukan dengan menghitung kuartil pertama (Q1) dan kuartil ketiga (Q3) dari data. Nilai IQR kemudian diperoleh dengan menggunakan rumus berikut:

**𝐼𝑄𝑅** = 𝑄3 − 𝑄1

Dimana:
𝑄1 adalah kuartil pertama
𝑄3 adalah kuartil ketiga

Berdasarkan nilai IQR ini, ditentukan batas bawah dan batas atas yang berfungsi sebagai ambang untuk mengidentifikasi nilai-nilai yang dianggap ekstrem atau menyimpang dari pola umum data. Data yang berada di luar rentang tersebut dikategorikan sebagai outlier dan dihapus dari dataset. Adapun rumus dari batas bawah dan batas atas adalah sebagai berikut:

**Batas atas** = Q3 + 1,5 × IQR 
**Batas Bawah** = Q1 - 1,5 × IQR

Setelah menerapkan metode IQR untuk menghilangkan outlier, sebanyak 26 data terdeteksi sebagai outlier dan dikeluarkan dari analisis. Akibatnya, jumlah data dalam dataset berkurang dari **1000 menjadi 974.**

Selanjutnya, dilakukan proses Principal Component Analysis (PCA) dengan tujuan mereduksi dimensi data serta mengatasi masalah multikolinearitas. Dalam hal ini, fitur 'Weight' dan 'BMI' digabungkan menjadi satu fitur baru bernama 'WeightBMICombined'. Penggabungan ini dilakukan karena kedua fitur tersebut memiliki nilai korelasi yang sangat tinggi, yang dapat menyebabkan informasi yang tumpang tindih dalam model. Setelah pembentukan fitur baru, fitur 'Weight' dan 'BMI' dihapus dari dataset untuk menghindari redundansi data.

![Gambar 4 analisis Principal Component Analysis (PCA)](https://github.com/user-attachments/assets/8292602a-fd09-4a99-90ac-623c23113dd0)

**Gambar 4 analisis Principal Component Analysis (PCA)**

Hasil pemetaan kategori ObesityCategory ke angka pada kolom ObesityCategory_num dengan aturan 'Underweight': 1, 'Normal weight': 2, 'Overweight': 3, dan 'Obese': 4 adalah sebagai berikut:

![Gambar 5 Hasil pemetaan kategori ObesityCategory ke angka pada kolom ObesityCategory_num](https://github.com/user-attachments/assets/0706f5dc-2318-4647-a38e-b55080418023)

**Gambar 5 Hasil pemetaan kategori ObesityCategory ke angka pada kolom ObesityCategory_num**

Hasil transformasi One-Hot Encoding tersebut dapat dilihat pada gambar berikut.

![Gambar 6 Hasil Transformasi One-Hot Encoding](https://github.com/user-attachments/assets/61d30bfc-b68c-4bc1-b157-eaac9631a2e0)

**Gambar 6 Hasil Transformasi One-Hot Encoding**

Pada proyek ini, digunakan metode Train Test Split dari library sklearn.model_selection untuk membagi dataset menjadi data latih dan data uji dengan perbandingan 90:10 serta menggunakan random_state sebesar 123 untuk menjaga konsistensi pembagian data. 

Selain itu, dilakukan proses standardisasi menggunakan StandardScaler dari library sklearn.preprocessing untuk menormalisasi dataset agar setiap fitur memiliki skala yang sama.

Semua proses ini dilakukan untuk memastikan model yang dibangun memiliki performa yang baik dan dapat menghasilkan prediksi yang akurat.

## Modeling

Dalam proyek ini, proses pemodelan dilakukan menggunakan tiga algoritma, yaitu:
1. Random Forest merupakan algoritma ensemble learning berbasis pohon keputusan yang membentuk banyak decision tree dan menggabungkan hasilnya untuk menghasilkan prediksi yang lebih akurat dan stabil. Pada kasus klasifikasi, algoritma ini bekerja dengan prinsip voting. Pada implementasi ini digunakan **parameter max_depth=20** untuk membatasi kedalaman maksimum pohon, guna mengurangi risiko overfitting. Adapun algoritma ini memiliki kelebihan dan kekurangan, sebagai berikut:
   
**Kelebihan:**
- Akurasi tinggi dan tahan terhadap overfitting.
- Dapat menangani data dengan fitur yang kompleks.
- Memberikan informasi pentingnya fitur (feature importance).
  
**Kekurangan:**
- Membutuhkan waktu komputasi lebih lama.
- Kurang interpretatif karena terdiri dari banyak pohon.

2. K-Nearest Neighbors (KNN) termasuk dalam instance-based learning, di mana proses klasifikasi dilakukan dengan menentukan kelas suatu data baru berdasarkan mayoritas kelas dari 'k' tetangga terdekatnya dalam ruang fitur. Pada implementasi ini digunakan model KNeighborsClassifier() dengan **parameter n_neighbors=5 dan weights='distance'**, yang menunjukkan bahwa algoritma akan mempertimbangkan 5 tetangga terdekat, sedangkan weights='distance' memberikan bobot lebih besar pada tetangga yang lebih dekat, sehingga mempengaruhi hasil prediksi secara proporsional berdasarkan jarak. Seperti algoritma lainnya, KNN memiliki kelebihan dan kekurangan, sebagai berikut:
   
**Kelebihan:**
- Sederhana dan mudah diimplementasikan.
- Tidak memerlukan proses pelatihan model.
  
**Kekurangan:**
- Performa lambat pada dataset besar.
- Sangat bergantung pada pemilihan nilai k dan skala fitur.

3. Support Vector Machine (SVM)  algoritma terakhir yang digunakan dalam prediksi klasifikasi tingkat obesitas. Algoritma ini merupakan algoritma klasifikasi yang bertujuan untuk mencari hyperplane terbaik yang dapat memisahkan dua atau lebih kelas dalam ruang fitur. Tujuannya adalah untuk memaksimalkan margin antara kelas-kelas tersebut agar model memiliki kemampuan generalisasi yang baik. Pada implementasi ini digunakan model SVC() dengan **parameter default**, yang secara otomatis menggunakan kernel RBF (Radial Basis Function). Seperti halnya algoritma Random Forest, SVM juga memiliki kelebihan dan kekurangan, sebagai berikut:
   
**Kelebihan:**
- Efektif untuk data berdimensi tinggi dan margin antar kelas yang jelas.
- Kuat terhadap overfitting dalam kondisi tertentu.
  
**Kekurangan:**
- Tidak cocok untuk dataset besar karena waktu komputasi tinggi.
- Sensitif terhadap outlier dan membutuhkan normalisasi data.

Berdasarkan hasil evaluasi akurasi, algoritma Random Forest dipilih sebagai model terbaik karena menunjukkan akurasi tertinggi dan performa yang paling stabil dibandingkan dengan SVM dan KNN. Random Forest mampu menangani kompleksitas data klasifikasi dengan baik, terutama dalam memprediksi tingkat obesitas yang terdiri dari empat kategori, yaitu Underweight, Normal weight, Overweight, dan Obese. Model ini bekerja dengan efektif menggunakan fitur-fitur seperti berat badan, tinggi badan, indeks massa tubuh (BMI), jenis kelamin, dan usia, sehingga menjadi solusi paling optimal dalam mengklasifikasikan level obesitas. Adapun tabel hasil evaluasi berdasarkan akurasi ditampilkan pada bagian Evaluasi.

## Evaluation

Pada tahap evaluasi, digunakan metrik akurasi (accuracy) sebagai ukuran kinerja model. Metrik ini digunakan untuk mengukur sejauh mana model mampu mengklasifikasikan data secara benar secara keseluruhan. 

Akurasi dihitung dengan rumus berikut:

**Accuracy = Jumlah prediksi benar/ Jumlah total prediksi × 100%**

Berikut adalah hasil evaluasi untuk masing-masing model.
**Tabel Hasil Akurasi**

| Model         | Accuracy   |
|---------------|------------|
| RandomForest  | 0.918367   |
| KNN           | 0.683673   |
| SVM           | 0.357143   |

![Gambar 7 Hasil Akurasi](https://github.com/user-attachments/assets/92ef91fc-c7e8-4409-a1b5-d4fd2ae3cd98)

**Gambar 7 Hasil Akurasi**

Dari hasil evaluasi tersebut, terlihat bahwa Random Forest memiliki akurasi paling tinggi dibandingkan KNN dan SVM, menjadikannya model terbaik dalam konteks klasifikasi tingkat obesitas pada dataset ini.

Evaluasi ini menunjukkan bahwa model Random Forest dapat memprediksi kategori obesitas dengan baik dan dapat digunakan sebagai alat bantu dalam mendeteksi risiko obesitas secara dini.

## Referensi
1. Aini, E. D. N., Khasanah, R. A., Ristyawan, A., & Diniati, E. (2024, July). Penggunaan Data Mining untuk Prediksi tingkat Obesitas di Meksiko Menggunakan Metode Random Forest. In Prosiding SEMNAS INOTEK (Seminar Nasional Inovasi Teknologi) (Vol. 8, No. 3, pp. 1256-1265).
2. Setiyani, L., Indahsari, A. N., & Roestam, R. (2023). Analisis Prediksi Level Obesitas Menggunakan Perbandingan Algoritma Machine Learning dan Deep Learning. JTERA (Jurnal Teknol. Rekayasa), 8(1), 139.
3. Sitanggang, D., & Sherly, S. (2022). Model Prediksi Obesitas dengan Menggunakan Support Vector Machine. Jurnal Sistem Informasi dan Ilmu Komputer, 5(2), 172-175.
