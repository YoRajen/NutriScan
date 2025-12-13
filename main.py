from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
import os

app = Flask(__name__)

# --- KONFIGURASI PATH (Gunakan Relative Path untuk Deploy) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
XLSX_PATH = os.path.join(BASE_DIR, "Kandungan_nutrisi.xlsx")
CLASSES_TXT = os.path.join(BASE_DIR, "classes.txt")

# --- LOAD MODEL & DATA ---
try:
    model = YOLO(MODEL_PATH)
    
    with open(CLASSES_TXT, "r", encoding="utf-8") as f:
        class_list = [x.strip() for x in f.read().splitlines() if x.strip()]

    # Load Excel dan normalisasi index
    nutrition_df = pd.read_excel(XLSX_PATH)
    # Pastikan kolom pertama jadi index dan string
    nutrition_df.set_index(nutrition_df.columns[0], inplace=True)
    nutrition_df.index = nutrition_df.index.astype(str).str.lower().str.strip()
    
    print("Model dan Data berhasil dimuat.")
except Exception as e:
    print(f"Error loading files: {e}")

@app.route('/')
def home():
    return render_template('index.html')

# --- BAGIAN FUNGSI DETECT ---
@app.route('/detect', methods=['POST'])
def detect():
    # 1. Validasi file upload
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    # 2. Baca data file dan konversi ke array
    try:
        # Membaca seluruh data file ke dalam byte stream
        img_bytes = file.read()
        
        # Mengonversi byte stream menjadi array NumPy
        np_arr = np.frombuffer(img_bytes, np.uint8)
        
        # Menggunakan OpenCV untuk mendecode array menjadi objek gambar (color)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    except Exception as e:
        print(f"Error reading or decoding file: {e}")
        # Jika terjadi error teknis saat membaca/decode
        return jsonify({'error': f'Gagal memproses file gambar. Error: {str(e)}'}), 500

    # 3. Pemeriksaan Kritis: Verifikasi apakah decoding berhasil
    # cv2.imdecode akan mengembalikan None jika format gambar tidak didukung atau korup
    if image is None:
        return jsonify({'error': 'Format gambar tidak valid atau korup (cv2.imdecode failed).'}), 400

    # 4. Prediksi YOLO (image sekarang dijamin berupa array NumPy yang valid)
    try:
        results = model.predict(image, imgsz=640, conf=0.25)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return jsonify({'error': f'Gagal menjalankan model AI. Error: {str(e)}'}), 500

    detected_foods = []
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = class_list[cls_id]
        
        # Inisialisasi nutrisi dan ukuran 0
        nutrisi = {
            "calories": 0, "carbs": 0, "protein": 0, 
            "fat": 0, "fiber": 0, "sugar": 0,
            "size": "-"
        }
        
        # Kunci Pencarian (tetap sama)
        lookup_key = name.lower().strip()
        
        found = False
        if lookup_key in nutrition_df.index:
            row = nutrition_df.loc[lookup_key]
            found = True
        else:
            alt_key = lookup_key.replace("_", " ")
            if alt_key in nutrition_df.index:
                row = nutrition_df.loc[alt_key]
                found = True

        if found:
            try:
                # MAPPING KOLOM (SESUAIKAN DENGAN NAMA KOLOM DI EXCEL ANDA)
                # Berdasarkan file CSV Anda: Nama, Kalori, Lemak, Karbohidrat, Protein, Serat, Gula, Ukuran
                # row.get(header_kolom, nilai_default)
                nutrisi = {
                    "calories": float(row.get('Kalori', 0)) if row.get('Kalori') is not None else 0,
                    "carbs": float(row.get('Karbohidrat', 0)) if row.get('Karbohidrat') is not None else 0,
                    "protein": float(row.get('Protein', 0)) if row.get('Protein') is not None else 0,
                    "fat": float(row.get('Lemak', 0)) if row.get('Lemak') is not None else 0,
                    "fiber": float(row.get('Serat', 0)) if row.get('Serat') is not None else 0,
                    "sugar": float(row.get('Gula', 0)) if row.get('Gula') is not None else 0,
                    # Pastikan Ukuran diambil sebagai string
                    "size": str(row.get('Ukuran', '-')).strip() if row.get('Ukuran') is not None else '-',
                }
            except Exception as e:
                print(f"Error converting data for {name}: {e}")

        detected_foods.append({
            "name": name,
            "foodKey": name.lower().replace(" ", "_"),
            **nutrisi
        })

    return jsonify(detected_foods)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)