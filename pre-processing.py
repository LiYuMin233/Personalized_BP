import math
import json
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import lightgbm as lgb
from tqdm import tqdm
import librosa
import matlab.engine
import csv
import numpy as np
import pandas as pd
import scipy.signal as signal
# 在环境变量中加入安装的Graphviz路径
import os

def band_pass_filter(original_signal, order, fc1,fc2, fs):
    '''
    中值滤波器
    :param original_signal: 音频数据
    :param order: 滤波器阶数
    :param fc1: 截止频率
    :param fc2: 截止频率
    :param fs: 音频采样率
    :return: 滤波后的音频数据
    '''
    b, a = signal.butter(N=order, Wn=[2*fc1/fs,2*fc2/fs], btype='bandpass')
    new_signal = signal.filtfilt(b, a, original_signal)
    return new_signal

def segment_raw_data(ECG, PPG, PCG):
    fs = 250
    window = 10
    seg = window * fs
    step = 10
    temp_ECG = []
    temp_PPG = []
    temp_PCG = []
    datalen = min(len(ECG),len(PPG),int(len(PCG)/4))
    # print(datalen)
    LEN = ((datalen // fs) - window) // step + 1
    print(f'数据分段: {LEN}')
    for f in range(0, LEN):
        if f + 1 == 1:
            ecgseg = ECG[0:(f + 1) * seg]
            ppgseg = PPG[0:(f + 1) * seg]
            pcgseg = PCG[0:(f + 1) * seg * 4]
        elif (f + 1 <= LEN) and (f + 1 > 1):
            ecgseg = ECG[f * fs * step: f * step * fs + seg]
            ppgseg = PPG[f * fs * step: f * step * fs + seg]
            pcgseg = PCG[f * fs * step * 4: f * step * fs * 4 + seg * 4]
        else:
            ecgseg = ECG[f * step * fs:datalen]
            ppgseg = PPG[f * step * fs:datalen]
            pcgseg = PCG[f * step * fs * 4:datalen*4]


        temp_ECG.append(ecgseg)
        temp_PPG.append(ppgseg)
        temp_PCG.append(pcgseg)
        # print(f'Signal Seg Index = {f+1}')

    E = np.array(temp_ECG)
    PP = np.array(temp_PPG)
    PC = np.array(temp_PCG)

    return E, PP, PC

def normalize(matrix, feature_min, feature_max, range_min=0, range_max=1):
    # 对矩阵进行列归一化
    normalized_matrix = (matrix - feature_min) / (feature_max - feature_min) * (range_max - range_min) + range_min
    return normalized_matrix


def read_data(file):
    signal = []
    if file[-11:] == 'ecg_all.txt':
        signal = np.loadtxt(file)
    elif file[-11:] == 'ppg_all.txt':
        signal = np.loadtxt(file)
    elif file[-11:] == 'pcg_all.txt':
        signal = np.loadtxt(file)
    return signal

def all_segmentsignas(path):

    raw_ECG = np.array(read_data(path + "ecg_all.txt"))
    raw_PPG = np.array(read_data(path + "ppg_all.txt"))
    raw_PCG = np.array(read_data(path + "pcg_all.txt"))
    PPG_orign = band_pass_filter(raw_PPG, 4, 0.5, 8, 250)
    ECG_orign = band_pass_filter(raw_ECG, 4, 0.5, 35, 250)
    PCG_orign = band_pass_filter(raw_PCG, 4, 20, 80, 1000)

    # 获取目录中的所有文件
    files = os.listdir(path)

    # 筛选出JSON文件
    json_files = [file for file in files if file.endswith('.json')]
    json_file_count = len(json_files)
    print(json_files[1])
    if json_file_count == 0:
        return "No JSON files found in the directory."
    # 读取数据内容
    if os.path.exists(path + 'user.json'):
        user_path = path + "user.json"
        with open(user_path, "r", encoding="utf-8") as f:
            user_info = json.loads(f.read())

        # with open(bp_path, "r", encoding="utf-8") as f:
        #     bp_label = json.loads(f.read())

        subject_id = user_info["name"]
        print(subject_id)
        Gen = user_info['sex']  # 性别
        years = user_info['age']  # 年龄
        weight = user_info['weight']  # 体重
        height = user_info['height'] * 100  # 身高
        inf_feature = np.hstack((Gen, years))
        inf_feature = np.hstack((inf_feature, weight))
        inf_feature = np.hstack((inf_feature, height))
        BMI = weight / ((height / 100) ** 2)
        inf_feature = np.hstack((inf_feature, BMI))

        with open(path + json_files[1], 'r', encoding='utf-8') as bp_file:
            # 使用json.load()方法解析JSON数据
            bp_label = json.load(bp_file)
            # 打印解析后的Python对象
        # print(bp_label)
        SBP = np.mean([label['sbp'] for label in bp_label if isinstance(label, dict) and 'sbp' in label])
        # print(SBP)
        DBP = np.mean([label['dbp'] for label in bp_label if isinstance(label, dict) and 'dbp' in label])
        # print(DBP)
        Date = os.path.splitext(json_files[1])[0]

    df_ppg = pd.read_csv('./model/ppg_min_max_values.csv')
    ppg_sqi_feature_min = df_ppg['min_vals'].values
    ppg_sqi_feature_max = df_ppg['max_vals'].values

    # 读取pcg特征归一化参数
    df_pcg = pd.read_csv('./model/pcg_min_max_values.csv')
    pcg_sqi_feature_min = df_pcg['min_vals'].values
    pcg_sqi_feature_max = df_pcg['max_vals'].values
    # plt.plot(PPG)
    # plt.show()
    # 信号质量特征计算
    PPG_sqi_features = []
    PPG_sqi_features_normalize = normalize(PPG_sqi_features.T, ppg_sqi_feature_min, ppg_sqi_feature_max)
    PCG_sqi_features = []
    PCG_sqi_features_normalize = normalize(PCG_sqi_features.T, pcg_sqi_feature_min, pcg_sqi_feature_max)

    # 信号质量模型
    # 加载模型
    load_ppg_model = lgb.Booster(model_file='./model/ppg_sqi_model.txt')
    load_pcg_model = lgb.Booster(model_file='./model/pcg_sqi_model.txt')
    # 模型预测
    ppg_sqi_pred = np.array(load_ppg_model.predict(PPG_sqi_features_normalize))
    pcg_sqi_pred = np.array(load_pcg_model.predict(PCG_sqi_features_normalize))
    print(ppg_sqi_pred, pcg_sqi_pred)
    # 筛选出信号质量较好的信号
    ECG, PPG, PCG = segment_raw_data(raw_ECG, raw_PPG, raw_PCG)
    good_ppg = -PPG[(ppg_sqi_pred == 0) & (pcg_sqi_pred == 0)]
    good_pcg = PCG[(ppg_sqi_pred == 0) & (pcg_sqi_pred == 0)]
    good_ecg = ECG[(ppg_sqi_pred == 0) & (pcg_sqi_pred == 0)]
    return inf_feature, SBP, DBP, good_ecg, good_ppg, good_pcg
