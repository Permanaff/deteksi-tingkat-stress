from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import io
import base64
from flask import send_file



app = Flask(__name__)
app.secret_key = 'Asdu2843$%j!4'



@app.route("/")
def home(): 
    return render_template('home.html')



@app.route("/deteksi-stress")
def deteksi_stress(): 
    return render_template('stress.html')



@app.route("/detect-stress", methods=["POST"]) 
def detect_stress() : 
    variabel_fuzzy = request.json

    kualitas_tidur = variabel_fuzzy['kualitas_tidur']
    performa_akademik = variabel_fuzzy['performa_akademik']
    hub_mhs_dosen = variabel_fuzzy['hub_mhs_dosen']
    support_sosial = variabel_fuzzy['support_sosial']
    kondisi_kehidupan = variabel_fuzzy['kondisi_kehidupan']

    # if len(variabel_fuzzy) != 5:
    #     return jsonify({'error': 'Data fitur tidak lengkap'}), 400
    
    stress_level = compute_stress_level(kualitas_tidur, performa_akademik, hub_mhs_dosen, support_sosial, kondisi_kehidupan)
    
    if stress_level < 50:
        tingkat_stress = "Normal"
    else:
         tingkat_stress = "Tinggi"

    keanggotaan = {
        "kualitas_tidur": two_category_membership(kualitas_tidur),
        "performa_akademik": academic_performance_membership(performa_akademik),
        "hub_mhs_dosen": two_category_membership(hub_mhs_dosen),
        "support_sosial": two_category_membership(support_sosial),
        "kondisi_kehidupan": two_category_membership(kondisi_kehidupan),
    }

    fuzzy_plot_base64 = plot_all_fuzzy_memberships()

    return jsonify({
        "level_stress": f"{stress_level:.2f}",
        "tingkat_stress": tingkat_stress,
        "keanggotaan": keanggotaan,
        "fuzzy_plot_base64": fuzzy_plot_base64  # bisa ditampilkan di frontend
    }), 200
    


# Fungsi untuk faktor dengan 2 kategori (Buruk, Baik)
def two_category_membership(x, low=30, high=70):
    if x <= low:
        return {'Buruk': 1, 'Baik': 0}
    elif x < high:
        return {'Buruk': (high - x)/(high - low), 'Baik': (x - low)/(high - low)}
    else:
        return {'Buruk': 0, 'Baik': 1}



# Fungsi untuk Academic Performance (Rendah, Sedang, Tinggi)
def academic_performance_membership(x, low=25, mid=50, high=75):
    if x <= low:
        return {'Rendah': 1, 'Sedang': 0, 'Tinggi': 0}
    elif x < mid:
        rendah = (mid - x)/(mid - low)
        sedang = (x - low)/(mid - low)
        return {'Rendah': rendah, 'Sedang': sedang, 'Tinggi': 0}
    elif x <= high:
        return {'Rendah': 0, 'Sedang': 1, 'Tinggi': 0}
    elif x < 80:
        sedang = (80 - x)/(high - mid)
        tinggi = (x - mid)/(high - mid)
        return {'Rendah': 0, 'Sedang': sedang, 'Tinggi': tinggi}
    else:
        return {'Rendah': 0, 'Sedang': 0, 'Tinggi': 1}  
    


def determine_consequent(rule):
    negative_count = 0
    if rule['Sleep Quality'] == 'Buruk':
        negative_count += 1
    if rule['Academic Performance'] == 'Rendah': 
        negative_count += 1
    if rule['Hubungan Dosen-Mahasiswa'] == 'Buruk':
        negative_count += 1
    if rule['Social Support'] == 'Buruk':
        negative_count += 1
    if rule['Living Condition'] == 'Buruk':
        negative_count += 1
    return 'Tinggi' if negative_count >= 3 else 'Normal'



def compute_stress_level(sleep_quality, academic_performance, hubungan, social_support, living_condition):
    input_dict = {
        'Sleep Quality': sleep_quality,
        'Academic Performance': academic_performance,
        'Hubungan Dosen-Mahasiswa': hubungan,
        'Social Support': social_support,
        'Living Condition': living_condition
    }
    
    # Hitung nilai keanggotaan
    memberships = {}
    for factor, value in input_dict.items():
        if factor == 'Academic Performance':
            memberships[factor] = academic_performance_membership(value)
        else:
            memberships[factor] = two_category_membership(value)
    
    # Proses inferensi Tsukamoto
    factor_names = list(input_dict.keys())
    category_lists = [list(memberships[factor].keys()) for factor in factor_names]
    
    total_alpha_z = 0
    total_alpha = 0
    for combo in itertools.product(*category_lists):
        mem_values = [memberships[factor_names[i]][combo[i]] for i in range(5)]
        alpha = min(mem_values)
        if alpha > 0:
            rule = {factor_names[i]: combo[i] for i in range(5)}
            consequent = determine_consequent(rule)
            z_i = 50 + 50 * alpha if consequent == 'Tinggi' else 50 - 50 * alpha
            total_alpha_z += alpha * z_i
            total_alpha += alpha
    
    return total_alpha_z / total_alpha if total_alpha > 0 else 50


def get_fuzzy_plot():
    x = np.linspace(0, 100, 1000)

    buruk = [two_category_membership(i)['Buruk'] for i in x]
    baik = [two_category_membership(i)['Baik'] for i in x]

    rendah = [academic_performance_membership(i)['Rendah'] for i in x]
    sedang = [academic_performance_membership(i)['Sedang'] for i in x]
    tinggi = [academic_performance_membership(i)['Tinggi'] for i in x]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x, buruk, label='Buruk', color='red')
    plt.plot(x, baik, label='Baik', color='green')
    plt.title('Fungsi Keanggotaan 2 Kategori')
    plt.xlabel('Nilai')
    plt.ylabel('Derajat Keanggotaan')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, rendah, label='Rendah', color='blue')
    plt.plot(x, sedang, label='Sedang', color='orange')
    plt.plot(x, tinggi, label='Tinggi', color='purple')
    plt.title('Fungsi Keanggotaan 3 Kategori (Akademik)')
    plt.xlabel('Nilai')
    plt.ylabel('Derajat Keanggotaan')
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

def plot_all_fuzzy_memberships():
    x = np.linspace(0, 100, 500)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Kualitas Tidur
    buruk = [two_category_membership(i)['Buruk'] for i in x]
    baik = [two_category_membership(i)['Baik'] for i in x]
    axes[0].plot(x, buruk, label='Buruk', color='red')
    axes[0].plot(x, baik, label='Baik', color='green')
    axes[0].set_title('Kualitas Tidur')
    axes[0].set_ylim(0, 1.1)

    # 2. Performa Akademik
    rendah = [academic_performance_membership(i)['Rendah'] for i in x]
    sedang = [academic_performance_membership(i)['Sedang'] for i in x]
    tinggi = [academic_performance_membership(i)['Tinggi'] for i in x]
    axes[1].plot(x, rendah, label='Rendah', color='blue')
    axes[1].plot(x, sedang, label='Sedang', color='orange')
    axes[1].plot(x, tinggi, label='Tinggi', color='purple')
    axes[1].set_title('Performa Akademik')
    axes[1].set_ylim(0, 1.1)

    # 3. Hubungan Mahasiswa-Dosen
    buruk = [two_category_membership(i)['Buruk'] for i in x]
    baik = [two_category_membership(i)['Baik'] for i in x]
    axes[2].plot(x, buruk, label='Buruk', color='red')
    axes[2].plot(x, baik, label='Baik', color='green')
    axes[2].set_title('Hubungan Mahasiswa-Dosen')
    axes[2].set_ylim(0, 1.1)

    # 4. Support Sosial
    buruk = [two_category_membership(i)['Buruk'] for i in x]
    baik = [two_category_membership(i)['Baik'] for i in x]
    axes[3].plot(x, buruk, label='Buruk', color='red')
    axes[3].plot(x, baik, label='Baik', color='green')
    axes[3].set_title('Support Sosial')
    axes[3].set_ylim(0, 1.1)

    # 5. Kondisi Kehidupan
    buruk = [two_category_membership(i)['Buruk'] for i in x]
    baik = [two_category_membership(i)['Baik'] for i in x]
    axes[4].plot(x, buruk, label='Buruk', color='red')
    axes[4].plot(x, baik, label='Baik', color='green')
    axes[4].set_title('Kondisi Kehidupan')
    axes[4].set_ylim(0, 1.1)


    # Tampilan
    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')






if __name__== "__main__":
    app.run(debug=True)