import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model Decision Tree
dt_model = joblib.load("lani.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None

    if request.method == "POST":
        kehadiran = float(request.form["Kehadiran_Persen"])
        tugas = float(request.form["Nilai_Tugas"])
        uts = float(request.form["Nilai_UTS"])
        uas = float(request.form["Nilai_UAS"])
        ipk = float(request.form["IPK"])

        data_baru = pd.DataFrame({
            'Kehadiran_Persen': [kehadiran],
            'Nilai_Tugas': [tugas],
            'Nilai_UTS': [uts],
            'Nilai_UAS': [uas],
            'IPK': [ipk]
        })

        prediksi = dt_model.predict(data_baru)

        hasil = "Lulus" if prediksi[0] == 1 or prediksi[0] == "Lulus" else "Tidak Lulus"

    return render_template("index.html", hasil=hasil)

if __name__ == "__main__":
    app.run(debug=True)
