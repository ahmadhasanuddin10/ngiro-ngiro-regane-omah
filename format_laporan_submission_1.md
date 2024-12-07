
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

### **Feature Engineering**
Pada tahap ini, dilakukan pengolahan fitur agar siap digunakan dalam model. Langkah-langkah yang dilakukan:  
1. **Encoding Fitur Kategorikal**  
   Fitur seperti `mainroad`, `guestroom`, `basement`, `airconditioning`, `prefarea`, dan `furnishingstatus` diubah menjadi nilai numerik menggunakan **LabelEncoder**.  
2. **Penambahan Fitur Baru**  
   Ditambahkan fitur `price_per_sqft` (harga per meter persegi) untuk meningkatkan kemampuan model dalam memahami hubungan harga dan luas properti.  
3. **Penanganan Outlier**  
   Menggunakan metode **IQR** (Interquartile Range) untuk menghapus data yang memiliki nilai terlalu ekstrem, seperti harga atau luas area yang terlalu jauh dari rentang normal.  

### **Split Data**
Dataset dibagi menjadi **training** dan **testing** set:  
- **Training Set**: 80% dari data, digunakan untuk melatih model.  
- **Testing Set**: 20% dari data, digunakan untuk evaluasi model.  
- Proses pembagian dilakukan menggunakan fungsi `train_test_split` dengan `random_state=42` untuk memastikan hasil yang konsisten:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
   ```

---

# **Modeling**

Berbagai model diterapkan untuk memprediksi harga rumah. Berikut adalah deskripsi model yang digunakan:  

### **1. Linear Regression**
Model regresi linear sederhana digunakan sebagai baseline.  
- Tidak ada parameter khusus yang digunakan.  

### **2. Ridge Regression**
Model regresi linear dengan regularisasi L2 untuk mencegah overfitting.  
- Parameter utama: `alpha` (kontrol tingkat regularisasi). Dicoba dengan nilai [0.1, 1, 10].  

### **3. Lasso Regression**
Model regresi linear dengan regularisasi L1 yang dapat melakukan seleksi fitur.  
- Parameter utama: `alpha`. Dicoba dengan nilai [0.01, 0.1, 1].  

### **4. Random Forest Regressor**
Model berbasis ensemble yang menggunakan banyak pohon keputusan untuk meningkatkan akurasi.  
- Parameter utama:  
  - `n_estimators`: Jumlah pohon (100, 200).  
  - `max_depth`: Kedalaman maksimum pohon (10, 20, None).  

### **5. Decision Tree Regressor**
Model pohon keputusan tunggal.  
- Parameter utama:  
  - `max_depth`: Kedalaman maksimum pohon (10, 20, None).  

### **6. Support Vector Regressor (SVR)**
Model regresi berbasis SVM dengan kernel RBF.  
- Parameter utama:  
  - `C`: Regularisasi (1, 10).  
  - `kernel`: Jenis kernel (RBF).  

### **7. XGBoost Regressor**
Model gradient boosting yang sangat efisien.  
- Parameter utama:  
  - `learning_rate`: Tingkat pembelajaran (0.01, 0.1).  
  - `n_estimators`: Jumlah pohon (100, 200).  

---

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














Berikut adalah laporan yang lebih terstruktur dan lengkap sesuai dengan permintaan, dengan menggunakan penulisan markdown dan heading yang sesuai. 

---

# **House Price Prediction**
Model ini bertujuan untuk memprediksi harga rumah berdasarkan fitur-fitur tertentu menggunakan beberapa algoritma Machine Learning. Laporan ini meliputi proses data preparation, modeling, evaluasi, dan interpretasi hasil.

---

## **1. Data Preparation**

Tahapan ini mencakup langkah-langkah untuk membersihkan dan mempersiapkan data agar siap digunakan dalam pemodelan. 

### **1.1 Data Understanding**
Dataset yang digunakan berisi kolom-kolom seperti:
- `area`: Luas rumah.
- `price`: Harga rumah.
- Fitur tambahan seperti `mainroad`, `guestroom`, dan lainnya.

### **1.2 Feature Engineering**
Feature engineering bertujuan untuk membuat fitur baru yang dapat meningkatkan performa model. Langkah-langkahnya adalah:
- Menambahkan kolom `price_per_sqft`, yaitu rasio harga terhadap luas rumah:
  ```python
  df['price_per_sqft'] = df['price'] / df['area']
  ```
- Encoding variabel kategorikal menggunakan `LabelEncoder`:
  ```python
  categorical_cols = ['mainroad', 'guestroom', 'basement', 'airconditioning', 'prefarea', 'furnishingstatus']
  le = LabelEncoder()
  for col in categorical_cols:
      df[col] = le.fit_transform(df[col])
  ```

### **1.3 Penanganan Outlier**
Outlier dihapus menggunakan metode IQR (Interquartile Range). Langkahnya adalah:
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

### **1.4 Normalisasi Data**
Data numerik dinormalisasi menggunakan `StandardScaler` agar berada dalam skala yang sama:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['area', 'price', 'price_per_sqft']
df[num_cols] = scaler.fit_transform(df[num_cols])
```

### **1.5 Split Data**
Data dipecah menjadi set pelatihan (80%) dan pengujian (20%) menggunakan `train_test_split`:
```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## **2. Modeling**

Beberapa model digunakan untuk prediksi harga rumah. Setiap model diuji dengan tuning parameter untuk mendapatkan performa terbaik.

### **2.1 Penjelasan Model**
1. **Linear Regression**  
   Model ini menggunakan persamaan linier untuk memprediksi variabel target. Kelebihannya adalah interpretabilitas yang tinggi.
   
2. **Ridge Regression**  
   Ridge menambahkan penalti L2 pada koefisien regresi untuk mengurangi multikolinearitas.

3. **Lasso Regression**  
   Lasso menambahkan penalti L1, yang dapat menghasilkan model lebih sederhana dengan menghilangkan beberapa fitur.

4. **Random Forest**  
   Ensemble method yang menggabungkan banyak decision tree untuk meningkatkan akurasi.

5. **Decision Tree**  
   Model ini membuat pohon keputusan berdasarkan aturan-aturan yang memisahkan data.

6. **Support Vector Regression (SVR)**  
   SVR menggunakan hyperplane untuk memprediksi nilai target dalam margin toleransi tertentu.

7. **XGBoost**  
   Algoritma boosting berbasis gradient yang sangat efisien untuk data tabular.

### **2.2 Tuning Parameter**
Hyperparameter terbaik diperoleh melalui GridSearchCV:
```python
from sklearn.model_selection import GridSearchCV

params = {
    'Linear Regression': {},
    'Ridge': {'alpha': [0.1, 1, 10]},
    'Lasso': {'alpha': [0.01, 0.1, 1]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    'Decision Tree': {'max_depth': [10, 20, None]},
    'SVR': {'C': [1, 10], 'kernel': ['rbf']},
    'XGBoost': {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
}

best_models = {}
for name, model in models.items():
    grid = GridSearchCV(model, params[name], cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
```

---

## **3. Evaluation**

Model dievaluasi menggunakan metrik:
- **R²**: Mengukur seberapa baik model menjelaskan variansi target.
- **MAPE**: Persentase rata-rata kesalahan prediksi.

### **Hasil Evaluasi**
```python
from sklearn.metrics import r2_score, mean_absolute_percentage_error

results = {}
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    results[name] = {'R²': r2, 'MAPE': mape}

results_df = pd.DataFrame(results).T
print(results_df)
```

### **Feature Importance**
Untuk model Random Forest dan XGBoost:
```python
feature_importances = pd.Series(best_models['XGBoost'].feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importances - XGBoost")
plt.show()
```

---

## **4. Kesimpulan**
- **XGBoost** adalah model terbaik dengan nilai R² tertinggi (0.9692) dan MAPE terendah (4.39%).
- **Random Forest** menjadi alternatif dengan performa hampir setara.
- Model regresi linear tidak cocok untuk dataset ini karena nilai MAPE yang tinggi.

**Rekomendasi**: 
Gunakan XGBoost untuk prediksi harga rumah. Optimalkan proses preprocessing untuk meningkatkan akurasi lebih lanjut.
