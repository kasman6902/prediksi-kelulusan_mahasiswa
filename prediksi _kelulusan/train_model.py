import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ==============================
# 1. Load Dataset
# ==============================
data = pd.read_csv("dataset_prediksi_kelulusan_mahasiswa.csv")

# ==============================
# 2. Pisahkan Fitur dan Target
# ==============================
X = data[['Kehadiran_Persen', 'Nilai_Tugas', 'Nilai_UTS', 'Nilai_UAS', 'IPK']]
y = data['Status_Kelulusan']   # 0 = Tidak Lulus, 1 = Lulus

# ==============================
# 3. Split Data Training & Testing
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# 4. Buat & Latih Model Decision Tree
# ==============================
dt_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

# ==============================
# 5. Simpan Model ke File .pkl
# ==============================
joblib.dump(dt_model, "lani.pkl")

print("Model Decision Tree BERHASIL disimpan ke lani.pkl")
print("Tipe isi lani.pkl:", type(dt_model))
