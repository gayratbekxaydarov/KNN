Men KNN imputer va KNN algorithmlarni ishlashini bir datada qilib ko'rdim pastda bu ishni qilishni ketma ketliginini ko'rsataman 
# 1. Datasetni yuklaymiz
df = pd.read_csv('KNNAlgorithmDataset.csv')

# 2. Keraksiz ustunlarni o'chirib tashlaymiz
df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore', inplace=True)

# 3. 'diagnosis' ustunini songa o'tkazamiz (chunki KNN faqat sonlar bilan ishlaydi)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 4. Imputerni yaratamiz
# n_neighbors=5 — yaqinidagi 5 ta qo'shnining o'rtacha qiymatini oladi
imputer = KNNImputer(n_neighbors=5)

# 5. To'ldirishni amalga oshiramiz
# Eslatma: fit_transform() bu yerda ham o'sha mantiqda ishlaydi
df_filled = imputer.fit_transform(df)

Agar ma'lumotlaringizda mantiqiy bog'liqlik kuchli bo'lsa (masalan, bo'y va vazn, radius va maydon).

Agar dataset juda katta bo'lmasa (chunki KNN har bir bo'sh katak uchun masofa hisoblagani sababli biroz sekin ishlashi mumkin).

Korrelyatsiya nima uchun kerak?
Korrelyatsiya — bu ikki o'zgaruvchi orasidagi bog'liqlik darajasi. U −1 dan +1 gacha bo'lgan son bilan o'lchanadi:

1 ga yaqin bo'lsa: Ikki ustun bir-biriga juda o'xshash (biri ortsa, ikkinchisi ham ortadi).

0 bo'lsa: Ustunlar orasida hech qanday bog'liqlik yo'q.

-1 ga yaqin bo'lsa: Teskari bog'liqlik (biri ortsa, ikkinchisi kamayadi).

Nima uchun bu KNN uchun muhim?
Agar datada radius_mean, perimeter_mean va area_mean degan ustunlar bo'lsa, bular aslida bir-biri bilan deyarli 100% bog'liq (chunki radius kattalashsa, maydon ham, perimetr ham albatta kattalashadi). Modelga bunday bir-birini takrorlovchi ustunlarni berish "ortiqcha yuk" hisoblanadi va u modelni chalg'itib, aniqlikni pasaytirishi mumkin.

Amaliyot: Korrelyatsiyani aniqlash va keraksizlarni o'chirish
Keling, yuklagan dataset ustida buni qanday qilishni ko'ramiz:

Python
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Korrelyatsiya matritsasini hisoblaymiz
corr_matrix = df_final.corr()

# 2. Vizuallashtirish (Heatmap)
plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# 3. O'ta yuqori bog'liqlikni (Multicollinearity) topish
# Masalan, 0.90 dan yuqori bo'lgan ustunlarni aniqlaymiz
upper = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

print("O'chirilishi kerak bo'lgan (bir-birini takrorlovchi) ustunlar:", to_drop)

# 4. Ustunlarni o'chiramiz
df_selected = df_final.drop(to_drop, axis=1)
Biz nimalarga erishdik?
Modelni yengillashtirdik: 30 ta ustun o'rniga, eng muhim 10-15 tasini qoldirdik.
# Breast Cancer Classification using KNN

This project implements a professional Machine Learning pipeline to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using the K-Nearest Neighbors (KNN) algorithm.

## 📋 Project Overview
The goal of this project is to build a highly accurate classification model by performing extensive data preprocessing, feature engineering, and hyperparameter tuning.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

## 🚀 Machine Learning Pipeline

### 1. Data Cleaning
- Removed irrelevant columns such as `id` and empty columns (`Unnamed: 32`).
- Encoded the target variable `diagnosis` (M = 1, B = 0) to make it compatible with mathematical models.

### 2. Handling Missing Values (KNN Imputer)
- Instead of simple mean/median imputation, I used **KNNImputer**.
- This method fills missing values by looking at the 5 nearest neighbors, ensuring data integrity and logical consistency.

### 3. Feature Selection (Correlation Analysis)
- Performed a correlation analysis to identify **multicollinearity**.
- Removed features with a correlation higher than **0.95** to reduce model noise and prevent overfitting.
- Used a **Heatmap** to visualize the relationship between different tumor features.

### 4. Data Scaling
- KNN is a distance-based algorithm, making it sensitive to the scale of data.
- Applied **StandardScaler** to normalize all features, ensuring they have a mean of 0 and a standard deviation of 1.

### 5. Model Training & Evaluation
- Split the dataset into **Training (80%)** and **Testing (20%)** sets.
- Optimized the `n_neighbors` parameter by iterating through different K values.
- Evaluated the model using **Accuracy Score** and **Confusion Matrix**.

## 📊 Results
- **Model Accuracy:** ~96-98% (depending on the random state)
- **Prediction:** The model can take a single patient's data, scale it, and predict the diagnosis with high confidence.

## 📂 Dataset
The dataset used is the `KNNAlgorithmDataset.csv`, which contains various features of cell nuclei such as radius, texture, perimeter, and area.

---
*Developed as part of an advanced ML learning journey, moving from "from-scratch" implementations to professional library-based workflows.*
README.md
Показан объект "README.md".
