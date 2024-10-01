from cgi import test
from flask import Flask, render_template, request, redirect, url_for
import pickle
from sklearn import preprocessing
import numpy as np
import os
import pandas as pd
from werkzeug.utils import secure_filename
import math
import json
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

app = Flask(__name__)

model = pickle.load(open("model2.pkl", "rb"))

# Set folder untuk upload file
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk halaman Model 
@app.route('/model')
def model_page():
    try:
        # Memuat data uji (sesuaikan dengan data aktual)
        test_data = pd.read_csv('Telkomsel_Data.csv')  # Ganti dengan path dataset sebenarnya

        # Mengganti nilai dengan benar
        test_data['Churn'] = test_data['Churn'].replace({'YA': 1, 'TIDAK': 0})
        test_data['Promosi Terakhir'] = test_data['Promosi Terakhir'].replace({'YA': 1, 'TIDAK': 0})

        # Pilih fitur yang akan digunakan
        feature_columns = ['Tagihan Rata-rata', 'Durasi Langganan (bulan)', 'Jumlah Panggilan', 'Penggunaan Data (GB)', 'Promosi Terakhir']
        test_features = test_data[feature_columns]
        y_true = test_data['Churn']  # Label sebenarnya

        # Normalisasi fitur
        test_features_normalized = normalize(test_features)

        # Pastikan model dilatih dengan jumlah fitur yang sesuai
        if test_features_normalized.shape[1] != model.n_features_in_:
            return "Jumlah fitur tidak sesuai dengan model. Model membutuhkan {} fitur, tetapi data uji memiliki {} fitur.".format(
                model.n_features_in_, test_features_normalized.shape[1])

        # Prediksi probabilitas churn
        y_proba = model.predict_proba(test_features_normalized)[:, 1]

        # Hitung ROC AUC score dan ROC curve
        roc_auc = roc_auc_score(y_true, y_proba)
        fpr, tpr, _ = roc_curve(y_true, y_proba)

        # Siapkan data untuk Chart.js
        roc_curve_data = {
            'fpr': fpr.tolist(),  # False Positive Rate
            'tpr': tpr.tolist()   # True Positive Rate
        }

        # Tampilkan hasil
        return render_template('model.html', predictions=y_proba, roc_auc=roc_auc, roc_curve_data=roc_curve_data)

    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}", 500
   

# Route untuk Prediction
@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

# Route untuk newData
@app.route('/new-data')
def new_data_page():
    # Cek apakah ada file CSV yang diunggah
    csv_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    if csv_files:
        # Inisialisasi DataFrame kosong
        combined_df = pd.DataFrame()
        
        # Baca setiap file CSV dan gabungkan ke dalam satu DataFrame
        for file in csv_files:
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, file))
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        combined_df['Churn'] = combined_df['Churn'].replace({1: 'YA', 0: 'TIDAK'})

        return render_template('new_data.html', df=combined_df)
    else:
        return render_template('new_data.html', message="Belum ada data.", df=None)



# Endpoint untuk upload file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.csv'):

        # Beri nama unik pada file yang diunggah
        filename = secure_filename(file.filename)
        unique_filename = f"{len(os.listdir(UPLOAD_FOLDER)) + 1}_{filename.rsplit('.', 1)[0]}.csv"
        
        # Simpan file ke folder uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        # Membaca file CSV menggunakan Pandas
        df = pd.read_csv(filepath)
        # Tampilkan isi file ke halaman web

        # Ganti nilai pada kolom "Promosi Terakhir"
        df['Promosi Terakhir'] = df['Promosi Terakhir'].replace({'YA': 1, 'TIDAK': 0})
        
        # Menyiapkan fitur untuk prediksi
        features = df[['Durasi Langganan (bulan)', 'Jumlah Panggilan', 'Penggunaan Data (GB)', 'Tagihan Rata-rata', 'Promosi Terakhir']].values

        # Lakukan normalisasi pada fitur
        features_normalized = preprocessing.normalize(features)
        
        # Lakukan prediksi dan tambahkan hasil prediksi sebagai kolom baru
        df['Churn'] = model.predict(features_normalized)
        
        # Ganti nilai kolom prediksi kembali ke bentuk aslinya
        df['Promosi Terakhir'] = df['Promosi Terakhir'].replace({1: 'YA', 0: 'TIDAK'})
        df['Churn'] = df['Churn'].replace({1: 'YA', 0: 'TIDAK'})
        
        # Simpan kembali file CSV yang telah diperbarui dengan kolom Churn
        updated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_filename}")
        df.to_csv(updated_filepath, index=False)

        return render_template('new_data.html', df=df,)
    return redirect(request.url)


# Kode untuk memprediksi pelanggan churn pada halaman Prediction
@app.route('/prediction', methods=['POST'])
def predict():
    try:
        # Ambil semua input dari form dengan validasi
        durasi = request.form.get('durasi')
        jml_panggilan = request.form.get('jml_panggilan')
        guna_data = request.form.get('guna_data')
        mean_tagihan = request.form.get('mean_tagihan')
        promosi = request.form.get('promosi')

        # Pastikan semua field diisi dan bisa dikonversi ke integer
        if not all([durasi, guna_data, jml_panggilan, mean_tagihan, promosi]):
            return render_template("prediction.html", error="Error: Semua field harus diisi.", prediction_text=None)
        
        # Konversi input ke integer
        durasi = int(durasi)
        jml_panggilan = int(jml_panggilan)
        guna_data = int(guna_data)
        mean_tagihan = int(mean_tagihan)
        promosi = int(promosi)
        
        # Masukkan input ke dalam array
        fitur = np.array([[durasi, jml_panggilan, guna_data, mean_tagihan, promosi]])

        new_data = preprocessing.normalize(fitur)
        
        # Lakukan prediksi
        prediction = model.predict(new_data)

        # Prediksi probabilitas churn
        prediction_proba = model.predict_proba(new_data)

        # Tampilkan hasil prediksi di tab "Prediction"
        return render_template("prediction.html", prediction_text=f"{'Churn' if prediction == 1 else 'Tidak Churn' if prediction == 0 else None}", churn=f"{math.floor(prediction_proba[0][1] * 100)}", tidak=f"{math.floor(prediction_proba[0][0] * 100)}" )
    
    except ValueError:
        return render_template("prediction.html", error="Error: Input harus berupa angka.", prediction_text=None)
    except Exception as e:
        return render_template("prediction.html", error=f"Error: {str(e)}", prediction_text=None)

# Route untuk newData
@app.route('/comparison')
def comparison_page():
    # Cek apakah ada file CSV yang diunggah
    csv_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    if csv_files:

        # data lama
        df = pd.read_csv('uploads/0_cth.csv')
        #kolom 'churn' yang berisi 1 untuk churn dan 0 untuk tidak churn
        df['Churn'] = df['Churn'].replace({'YA': 1, 'TIDAK': 0})

        iya = df['Churn'].sum()  # Jumlah churn
        tidak = len(df) - iya  # Jumlah tidak churn

        # Hitung persentase
        total = len(df)
        yaLama = round((iya / total) * 100, 2)  # Persentase churn
        tidakLama = round((tidak / total) * 100, 2)  # Persentase tidak churn


        # gabungan setelah upload data baru
        # Inisialisasi DataFrame
        combined_df = pd.DataFrame()
        
        # Baca setiap file CSV dan gabungkan ke dalam satu DataFrame
        for file in csv_files:
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, file))
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        #kolom 'churn' yang berisi 1 untuk churn dan 0 untuk tidak churn
        combined_df['Churn'] = combined_df['Churn'].replace({'YA': 1, 'TIDAK': 0})

        churn_count = combined_df['Churn'].sum()  # Jumlah churn
        not_churn_count = len(combined_df) - churn_count  # Jumlah tidak churn

        # Hitung persentase
        total = len(combined_df)
        yaBaru = round((churn_count / total) * 100, 2)  # Persentase churn
        tidakBaru = round((not_churn_count / total) * 100, 2)  # Persentase tidak churn


        # data yang baru diupload
        # Dapatkan file terbaru berdasarkan waktu modifikasi
        latest_df = max([os.path.join(UPLOAD_FOLDER, f) for f in csv_files], key=os.path.getmtime)

        df_new = pd.read_csv(latest_df)
        #kolom 'churn' yang berisi 1 untuk churn dan 0 untuk tidak churn
        df_new['Churn'] = df_new['Churn'].replace({'YA': 1, 'TIDAK': 0})

        jumlahIya = df_new['Churn'].sum()  # Jumlah churn
        jumlahTidak = len(df_new) - jumlahIya  # Jumlah tidak churn

        return render_template('comparison.html', df=combined_df, ya_baru=yaBaru, tidak_baru=tidakBaru, ya_lama=yaLama, tidak_lama=tidakLama, jumlah_iya=jumlahIya, jumlah_tidak=jumlahTidak)
    else:
        return render_template('comparison.html', message="Belum ada data.", df=None)

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)