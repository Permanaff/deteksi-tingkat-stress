from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import io
import base64



app = Flask(__name__)
app.secret_key = 'Asdu2843$%j!4'



@app.route("/")
def home(): 
    return render_template('home.html')



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
    
    if stress_level < 35:
        tingkat_stress = "Normal"
    elif stress_level < 65:
        tingkat_stress = "Sedang"
    else:
        tingkat_stress = "Tinggi"

    return jsonify({"level_stress" : f"{stress_level:.2f}", "tingkat_stress" : tingkat_stress}), 200
    


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
        rendah = (mid - x) / (mid - low)
        sedang = (x - low) / (mid - low)
        return {'Rendah': rendah, 'Sedang': sedang, 'Tinggi': 0}
    elif x < high:
        sedang = (high - x) / (high - mid)
        tinggi = (x - mid) / (high - mid)
        return {'Rendah': 0, 'Sedang': sedang, 'Tinggi': tinggi}
    else:
        return {'Rendah': 0, 'Sedang': 0, 'Tinggi': 1} 
    


def determine_consequent(rule):
    negative_count = 0
    if rule['Kualitas Tidur'] == 'Buruk':
        negative_count += 1
    if rule['Performa Akademik'] in ['Rendah', 'Sedang']:
        negative_count += 1
    if rule['Hubungan Dosen-Mhs'] == 'Buruk':
        negative_count += 1
    if rule['Support Sosial'] == 'Buruk':
        negative_count += 1
    if rule['Kondisi Kehidupan'] == 'Buruk':
        negative_count += 1

    if negative_count >= 4:
        return 'Tinggi'
    elif negative_count >= 2:
        return 'Sedang'
    else:
        return 'Normal'


def compute_stress_level(sleep_quality, academic_performance, hubungan, social_support, living_condition):
    input_dict = {
        'Kualitas Tidur': sleep_quality,
        'Performa Akademik': academic_performance,
        'Hubungan Dosen-Mhs': hubungan,
        'Support Sosial': social_support,
        'Kondisi Kehidupan': living_condition
    }
    
    # Hitung nilai keanggotaan
    memberships = {}
    for factor, value in input_dict.items():
        if factor == 'Performa Akademik':
            memberships[factor] = academic_performance_membership(value)
        else:
            memberships[factor] = two_category_membership(value)
    
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
            # Z nilai output fuzzy (misal: Normal=25, Sedang=50, Tinggi=75, adjusted dengan alpha)
            if consequent == 'Tinggi':
                z_i = 75 * alpha
            elif consequent == 'Sedang':
                z_i = 50 * alpha
            else:
                z_i = 25 * alpha
            total_alpha_z += alpha * z_i
            total_alpha += alpha
    
    return total_alpha_z / total_alpha if total_alpha > 0 else 50



if __name__== "__main__":
    app.run(debug=True)