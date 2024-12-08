
# **Prediksi Harga Properti - Ahmad Hasanuddin**

## **Domain Proyek**
**Latar Belakang:**  
Prediksi harga properti merupakan salah satu tantangan utama dalam industri real estate. Harga properti sangat dipengaruhi oleh berbagai faktor, seperti lokasi, fasilitas, luas bangunan, dan kondisi pasar, yang sering kali bersifat dinamis dan sulit dipahami tanpa analisis yang tepat. Pengambilan keputusan berdasarkan intuisi atau pengalaman semata cenderung menghasilkan ketidakakuratan, yang dapat berdampak negatif pada pembeli, penjual, maupun investor. Ada beberapa permasalahan yang dihadapi:
1. Harga properti dipengaruhi oleh banyak variabel kompleks, termasuk faktor fisik (luas bangunan, jumlah kamar), fasilitas (parkir, AC, basement), dan preferensi lokasi (akses jalan utama, daerah premium).  
2. Tren pasar real estate yang fluktuatif memerlukan pendekatan analisis yang lebih modern dan berbasis data untuk memprediksi nilai properti dengan akurat.  
3. Keputusan yang buruk dalam membeli atau menjual properti dapat menyebabkan kerugian finansial yang signifikan.

**Tujuan:**  
Membangun sistem prediksi harga properti berbasis machine learning yang dapat membantu para pelaku pasar real estate dalam:  
- Menentukan nilai properti yang wajar dan realistis.  
- Mengoptimalkan strategi investasi berdasarkan data yang akurat.  

**Permasalahan:**  
1. Bagaimana memanfaatkan fitur-fitur properti (seperti luas bangunan, jumlah kamar, dan lokasi) untuk memprediksi harga properti dengan akurat?  
2. Bagaimana meningkatkan akurasi prediksi dengan memilih algoritma machine learning yang tepat dan melakukan optimalisasi model?  

**Dampak:**  
- **Bagi Pembeli:** Membantu mereka memahami nilai properti yang diinginkan, sehingga dapat membuat keputusan pembelian yang lebih informatif.  
- **Bagi Penjual:** Memberikan harga jual yang kompetitif dan realistis berdasarkan kondisi pasar.  
- **Bagi Investor:** Mendukung strategi investasi yang berbasis data untuk memaksimalkan keuntungan.  

**Solusi yang Diusulkan:**  
Menerapkan algoritma machine learning, seperti Random Forest, XGBoost, dan Linear Regression, dengan optimalisasi parameter untuk meningkatkan akurasi prediksi. Model ini akan dirancang untuk memahami hubungan kompleks antar fitur dalam dataset properti dan menghasilkan prediksi harga yang akurat dan andal.

**Referensi:**
1. [DETERMINAN HARGA RUMAH DI INDONESIA](https://jurnal.uns.ac.id/dinamika/article/download/45934/28895)
2. [Survei Harga Properti Residensial BI](https://www.bi.go.id/id/publikasi/laporan/Documents/SHPR_Tw_I_2024.pdf)

---

## **Business Understanding**
### **Problem Statements**
1. Bagaimana memprediksi harga properti berdasarkan fitur seperti luas bangunan, jumlah kamar tidur, fasilitas, dan lokasi?
2. Bagaimana meningkatkan akurasi model prediksi harga properti dengan menggunakan algoritma machine learning yang tepat?

### **Goals**
1. Menghasilkan model machine learning yang mampu memprediksi harga properti berdasarkan fitur seperti luas bangunan, jumlah kamar tidur, fasilitas, dan lokasi?
2. Meningkatkan akurasi model prediksi harga properti dengan menggunakan algoritma machine learning yang tepa.

### **Solution Statements**
- **Model yang Digunakan:** 
  Menggunakan algoritma seperti Random Forest Regressor, XGBoost, dan Linear Regression untuk memodelkan data.
- **Optimalisasi:** 
  Hyperparameter tuning dengan GridSearchCV untuk menemukan konfigurasi parameter terbaik, seperti `n_estimators` dan `max_depth`.

---

## **Data Understanding**
Dataset yang digunakan berisi lebih dari 500 baris data mengenai properti, mencakup harga, luas bangunan, jumlah kamar, dan fasilitas lainnya. Dataset diunduh dari [Housing Price Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction).
| **Indikator**       | **Jumlah**                                      |
|-----------------------|----------------------------------------------------|
| `Jumlah Data`                 |545 |
| `Jumlah Kolom`                |13  |
| `Missing Value`               |0   |
| `Duplikat`                    |0   |

### **Deskripsi Fitur**
| **Nama Fitur**       | **Deskripsi**                                      |
|-----------------------|----------------------------------------------------|
| `price`              | Harga properti (target variabel).                  |
| `area`               | Luas bangunan dalam sqft.                          |
| `bedrooms`           | Jumlah kamar tidur.                                |
| `bathrooms`          | Jumlah kamar mandi.                                |
| `stories`            | Jumlah lantai.                                     |
| `mainroad`           | Properti di jalan utama (1 = Ya, 0 = Tidak).       |
| `guestroom`          | Properti memiliki ruang tamu (1 = Ya, 0 = Tidak).  |
| `basement`           | Properti memiliki ruang bawah tanah (1 = Ya, 0 = Tidak). |
| `airconditioning`    | Properti memiliki pendingin udara (1 = Ya, 0 = Tidak). |
| `parking`            | Jumlah tempat parkir.                              |
| `prefarea`           | Properti di daerah preferensi (1 = Ya, 0 = Tidak). |
| `furnishingstatus`   | Status furnitur (0 = Tidak Furnitur, 1 = Semi-Furnitur, 2 = Furnitur Lengkap). |
| `hotwaterheating`   |  properti memiliki fasilitas pemanas air  (1 = Ya, 0 = Tidak). |

---

# **Data Preparation**

---

### **Label Encoding untuk Variabel Kategorikal**
Variabel kategorikal yang berupa nilai non-numerik (misalnya `'yes'`/`'no'` atau kategori lainnya) dikonversi menjadi nilai numerik dengan menggunakan `LabelEncoder`. Variabel-variabel yang diubah meliputi:  
`['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']`.

Contoh hasil:  
- `'yes'` → 1  
- `'no'` → 0  

Kode:  
```python
encoder = LabelEncoder()
for col in label_cols:
    df[col] = encoder.fit_transform(df[col])
```

---

### **Normalisasi Kolom Numerik**
Kolom numerik `'area'` dan `'price'` dinormalisasi menggunakan `MinMaxScaler` agar nilai-nilai berada dalam rentang [0, 1]. Ini penting untuk menyamakan skala data dan mencegah bias pada algoritma yang sensitif terhadap skala.

Kode:  
```python
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
```

---

### **Penanganan Outliers dengan IQR**
Untuk kolom `'price'`, `'area'`, dan `'price_per_sqft'`, data yang merupakan *outliers* (berada di luar rentang [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]) dihapus.  
Metode IQR (Interquartile Range) membantu mengurangi pengaruh data ekstrem.

Kode:  
```python
def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

outlier_cols = ['price', 'area', 'price_per_sqft']
df = remove_outliers(df, outlier_cols)
```

---

### **Penambahan Fitur Baru**
Fitur `'price_per_sqft'` ditambahkan untuk memberikan informasi harga per satuan luas area (`price / area`). Fitur ini berguna untuk analisis lebih lanjut.

Kode:  
```python
df['price_per_sqft'] = df['price'] / df['area']
```

---
#### **Mapping Kolom 'hotwaterheating'**
Kolom `'hotwaterheating'` yang berisi nilai kategorikal (`'yes'` atau `'no'`) diubah menjadi nilai numerik menggunakan *dictionary mapping*:
- `'yes'` → 1  
- `'no'` → 0  

Kode:
```python
hotwater_mapping = {'yes': 1, 'no': 0}
X['hotwaterheating'] = X['hotwaterheating'].map(hotwater_mapping)
```
### **Penanganan Missing Values**
Setelah encoding dan transformasi data, kolom yang masih memiliki nilai *missing* (`NaN`) diperiksa.  
Nilai *missing* diisi dengan **0** untuk mencegah error selama pemrosesan lebih lanjut.

Kode:  
```python
X.fillna(0, inplace=True)
```

---

### **Standardisasi Data**
Semua kolom dalam dataset dinormalisasi menggunakan `StandardScaler` sehingga memiliki mean = 0 dan standar deviasi = 1.  
Langkah ini memastikan setiap fitur memiliki kontribusi yang seimbang selama pelatihan model.

Kode:  
```python
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
```

---

### **Pembagian Data (Train-Test Split)**
Dataset dibagi menjadi data *training* (80%) dan *testing* (20%) menggunakan `train_test_split`.  
Parameter `random_state=42` memastikan pembagian yang konsisten saat dijalankan ulang.

Kode:  
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

---
Proses *preprocessing* ini mencakup:
1. **Encoding** variabel kategorikal.
2. **Normalisasi** dan **standardisasi** fitur numerik.
3. Penanganan **outliers** dan **missing values**.
4. Penambahan fitur baru untuk analisis (*feature engineering*).
5. Pembagian data untuk pelatihan dan pengujian model.

# **Modeling**

Berbagai model diterapkan untuk memprediksi harga rumah. Berikut adalah deskripsi model yang digunakan:  

### **1. Linear Regression**
Model regresi linear sederhana digunakan sebagai baseline.  
Linear Regression adalah model regresi sederhana yang mengasumsikan hubungan linear antara variabel independen (fitur) dan variabel dependen (target). Model ini bekerja dengan mencari garis terbaik yang meminimalkan jumlah kuadrat dari selisih antara nilai prediksi dan nilai sebenarnya.
- Tidak ada parameter khusus yang digunakan.  

### **2. Ridge Regression**
Ridge Regression adalah varian dari Linear Regression yang menambahkan penalti terhadap besarnya koefisien model untuk mengurangi overfitting. Penalti dihitung menggunakan **L2 regularization**, yaitu jumlah kuadrat dari nilai koefisien.
- Parameter utama: `alpha` (kontrol tingkat regularisasi). Dicoba dengan nilai [0.1, 1, 10].  

### **3. Lasso Regression**
Lasso Regression juga merupakan varian Linear Regression, tetapi menggunakan **L1 regularization**, yang menambahkan penalti terhadap nilai absolut dari koefisien. Model ini dapat menyetel beberapa koefisien menjadi nol, sehingga berguna untuk seleksi fitur.
  
- Parameter utama: `alpha`. Dicoba dengan nilai [0.01, 0.1, 1].  

### **4. Random Forest Regressor**
Random Forest adalah model ensambel berbasis pohon keputusan yang menggabungkan prediksi dari banyak pohon untuk meningkatkan akurasi dan mengurangi overfitting. Model ini memecah data menjadi beberapa subset dan membangun pohon keputusan independen pada masing-masing subset.  
- Parameter utama:  
  - `n_estimators`: Jumlah pohon (100, 200).  
  - `max_depth`: Kedalaman maksimum pohon (10, 20, None).  

### **5. Decision Tree Regressor**
Decision Tree bekerja dengan membagi data menjadi subset berdasarkan kondisi tertentu pada fitur, menghasilkan struktur pohon yang merepresentasikan aturan prediksi. Namun, model ini rentan terhadap overfitting pada data latih.  
- Parameter utama:  
  - `max_depth`: Kedalaman maksimum pohon (10, 20, None).  

### **6. Support Vector Regressor (SVR)**
SVR adalah model berbasis margin yang mencoba menemukan hyperplane terbaik untuk memprediksi nilai target dengan meminimalkan error dalam margin tertentu. Kernel RBF (Radial Basis Function) sering digunakan untuk menangani data non-linear. 
- Parameter utama:  
  - `C`: Regularisasi (1, 10).  
  - `kernel`: Jenis kernel (RBF).  

### **7. XGBoost Regressor**
XGBoost (Extreme Gradient Boosting) adalah model boosting berbasis pohon yang sangat efisien dan mampu menangani data tidak seimbang dengan baik. Model ini bekerja dengan membangun pohon secara berurutan, di mana setiap pohon baru mencoba memperbaiki kesalahan pohon sebelumnya.
- Parameter utama:  
  - `learning_rate`: Tingkat pembelajaran (0.01, 0.1).  
  - `n_estimators`: Jumlah pohon (100, 200).  

---

### **Hasil Terbaik:**
- **Model terbaik** berdasarkan nilai *R²* dan *MAPE* adalah **XGBoost Regressor**, dengan performa berikut:
  - **R²**: 0.9692
  - **MAPE**: 4.3856e+13  

Model ini memiliki kombinasi parameter terbaik:
- `learning_rate`: 0.1  
- `n_estimators`: 200

# **Evaluation**

### **Metode Evaluasi**
- **R² (R-squared)**: Mengukur seberapa baik model menjelaskan variansi data target.  
- **MAPE (Mean Absolute Percentage Error)**: Mengukur kesalahan prediksi relatif terhadap nilai sebenarnya dalam persentase.  

### **Hasil Evaluasi Model**

| **Model**            | **R²**    | **MAPE**              |
|-----------------------|-----------|-----------------------|
| Linear Regression     | 0.8792    | 1.543798e+14          |
| Ridge Regression      | 0.8791    | 1.537867e+14          |
| Lasso Regression      | 0.8536    | 7.150074e+12          |
| Random Forest         | 0.9686    | 9.720269e+13          |
| Decision Tree         | 0.9414    | 1.563750e+14          |
| SVR                  | 0.7285    | 5.871988e+14          |
| XGBoost               | 0.9692    | 4.385626e+13          |

### **Analisis Hasil**
1. **XGBoost**: Model terbaik dengan **R² tertinggi (0.9692)** dan **MAPE terendah (4.385626e+13)**.  
2. **Random Forest**: Alternatif yang baik dengan performa mendekati XGBoost.  
3. **SVR**: Model terburuk dengan **R² rendah (0.7285)** dan **MAPE tinggi**.  

### **Kesimpulan**

1. **Prediksi Harga Properti**
   Berdasarkan analisis dan pemodelan yang dilakukan, harga properti dapat diprediksi dengan cukup akurat menggunakan berbagai fitur, seperti **luas bangunan**, **jumlah kamar tidur**, **fasilitas** (seperti akses jalan, ruang tamu, basement, dll.), dan **lokasi** (termasuk keberadaan fasilitas seperti air conditioning dan furnitur). Dalam model yang dibangun, fitur-fitur ini diolah dan dipersiapkan dengan hati-hati melalui **encoding** untuk variabel kategorikal, **normalisasi** untuk data numerik, dan **penanganan outlier** untuk memastikan kualitas data.

   - **Hubungan antara luas bangunan dan harga** menunjukkan korelasi positif yang kuat, di mana semakin besar luas properti, semakin tinggi harga jualnya.
   - **Fasilitas** dan **lokasi** juga memberikan dampak signifikan pada harga properti, dengan faktor-faktor seperti ketersediaan ruang tamu, basement, dan status furnitur mempengaruhi harga rumah.

2. **Peningkatan Akurasi Model**
   Untuk meningkatkan akurasi prediksi harga properti, berbagai **algoritma machine learning** telah diterapkan. Dari hasil evaluasi, **XGBoost** dan **Random Forest** terbukti menjadi model yang paling akurat, dengan skor **R² tertinggi** (0.9692 untuk XGBoost dan 0.9686 untuk Random Forest), yang berarti model-model ini mampu menjelaskan hampir seluruh variansi data harga rumah.  
   
   - **XGBoost** menghasilkan performa terbaik dengan **MAPE terendah**, menunjukkan kesalahan prediksi yang paling kecil dibandingkan dengan model lainnya.
   - **Random Forest** juga menunjukkan hasil yang sangat baik dengan **R² tinggi**, meskipun sedikit lebih rendah daripada XGBoost.

   Sebaliknya, model-model seperti **SVR** dan **Linear Regression** memberikan hasil yang kurang memuaskan, dengan **R² rendah** dan **MAPE tinggi**, yang mengindikasikan bahwa model-model tersebut tidak cocok untuk dataset ini.

### **Rekomendasi**
- Untuk prediksi harga properti yang lebih akurat, **XGBoost** atau **Random Forest** adalah pilihan terbaik karena kedua model ini memberikan kinerja yang sangat baik.
- Meskipun **Linear Regression** dan **Ridge Regression** bisa digunakan sebagai baseline, model tersebut tidak dapat memberikan hasil yang optimal untuk dataset ini.
Dengan menggunakan algoritma yang tepat dan melakukan **preprocessing** yang baik, kita dapat memprediksi harga properti secara lebih akurat, yang dapat membantu para pengembang properti, investor, dan pembeli untuk membuat keputusan yang lebih informasional.
---

_Laporan ini telah disusun untuk memenuhi standar dokumentasi proyek machine learning yang rapi dan informatif. Untuk pertanyaan atau saran, silakan menghubungi._ 😊



___Semoga Tugas Di terima dengan Baik, mohon bimbingannya__







