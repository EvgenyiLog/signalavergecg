#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from  time_slice import time_slice
from read_edf import read_edf
from findpeaks import findpeaks
from peaksfind import peaksfind
from scipy import signal
import matplotlib.pyplot as plt
import biosppy
import numpy as np
from signalavergedecg import signalavergedecg
from compute_late_potentials_from_avg import compute_late_potentials_from_avg
from waveletscaleaogram import waveletscaleaogram
from compare_signals_mannwhitney import compare_signals_mannwhitney
import os
from signaladd import signaladd
from iqr_winsorize import iqr_winsorize
import pprint
from tqdm import tqdm 
from scipy import stats
from periodogram_power import periodogram_power
from plot_fixed_bin_histogram import plot_fixed_bin_histogram
from plot_histogram import plot_histogram
from calculate_skew_kurtosis import calculate_skew_kurtosis
from calculate_band_power_ratio import calculate_band_power_ratio
from  apply_mannwhitney_to_all import apply_mannwhitney_to_all
from rocauccurveml import  prepare_rf_data,train_rf_and_plot_roc
from iircombfilter import iircombfilter
import pandas as pd
import pprint
def main():
    save_dir_pic = "result/picture"
    os.makedirs(save_dir_pic, exist_ok=True)
    save_dir_txt = "result/txt"
    os.makedirs(save_dir_txt, exist_ok=True)
    save_dir_xlsx = "result/xlsx"
    os.makedirs(save_dir_xlsx, exist_ok=True)
    chanel_d,time,fs1,anot=read_edf("12_2_Not_filtered.edf","D:/ECG_IAI_RAS/RAT_NEW/12/2_rat/",[0, 1, 2, 3, 4, 5])
    sig1=chanel_d.get('II_LF           ')
    lf_sig,hf_sig,sig1,fs1,anot=signaladd("12_2_Not_filtered.edf","D:/ECG_IAI_RAS/RAT_NEW/12/2_rat/")
    sig1=iqr_winsorize(sig1,5)
    sig1=iircombfilter(sig1,fs1)

    print('Длительность,с')
    print(len(sig1)/fs1)
    print(anot)

    if not anot:
        stabilization_time=0
        ishemia_time=1688
        reperfusion_time=3600
    else:
        stabilization_time=0
        ishemia_time=next((k for k, v in anot.items() if v == 'Ishemia'), 0)
        reperfusion_time=next((k for k, v in anot.items() if v == 'Reperfusion'), 0)

    ynorm=time_slice(
        sig1 - np.mean(sig1),
        stabilization_time+5,
        stabilization_time+34,
        fs1
    )

    ypat=time_slice(
        sig1 - np.mean(sig1),
        ishemia_time,
        ishemia_time+34,
        fs1
    )

    plt.figure(figsize=(15,7))
    plt.plot(np.linspace(0,len(ynorm)/fs1,len(ynorm)),ynorm)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir_pic, "ecg_signal_norm.jpeg"), dpi=300)

    plt.figure(figsize=(15,7))
    plt.plot(np.linspace(0,len(ypat)/fs1,len(ypat)),ypat)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir_pic, "ecg_signal_pat.jpeg"), dpi=300)
    rpeaks=findpeaks(ynorm,fs1)
    saecgnorm=signalavergedecg(ynorm,fs1,rpeaks)
    rpeaksnorm,qpeaksnorm,speaksnorm,qrsonnorm,qrsoffnorm=peaksfind(saecgnorm,fs1)
    plt.figure(figsize=(15,7))
    plt.plot(np.linspace(0,len(saecgnorm)/fs1,len(saecgnorm)),saecgnorm)
    plt.plot(np.linspace(0,len(saecgnorm)/fs1,len(saecgnorm))[rpeaksnorm],saecgnorm[rpeaksnorm],'ro')
    plt.plot(np.linspace(0,len(saecgnorm)/fs1,len(saecgnorm))[speaksnorm],saecgnorm[speaksnorm],'bo')
    plt.plot(np.linspace(0,len(saecgnorm)/fs1,len(saecgnorm))[qpeaksnorm],saecgnorm[qpeaksnorm],'mo')
    plt.plot(np.linspace(0,len(saecgnorm)/fs1,len(saecgnorm))[qrsonnorm],saecgnorm[qrsonnorm],'k*')
    plt.plot(np.linspace(0,len(saecgnorm)/fs1,len(saecgnorm))[qrsoffnorm],saecgnorm[qrsoffnorm],'g*')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir_pic, "ecg_signal_averged_norm_peaks.jpeg"), dpi=300)

    plt.figure(figsize=(15,7))
    plt.plot(np.linspace(0,len(saecgnorm)/fs1,len(saecgnorm)),saecgnorm)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir_pic, "ecg_signal_averged_norm.jpeg"), dpi=300)
    parnorm=compute_late_potentials_from_avg(saecgnorm,fs1)


    bin_edgesnorm, bin_heightsnorm=plot_histogram(saecgnorm, save_path=os.path.join(save_dir_pic, "ecg_signal_averged_hist_norm.jpeg"))
    statsmomentsnorm=calculate_skew_kurtosis(saecgnorm)


    total_power_norm,freq_norm, Pxx_norm=periodogram_power(saecgnorm,fs1)
    bin_centersnorm, bin_powersnorm=plot_fixed_bin_histogram(freq_norm, Pxx_norm,save_path=os.path.join(save_dir_pic, "ecg_signal_averged_hist_psd_norm.jpeg"))
    plt.figure(figsize=(15,7))
    plt.plot(freq_norm, Pxx_norm)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir_pic, "ecg_signal_averged_psd_norm.jpeg"), dpi=300)


    Spnorm,frequenciesnorm,powernorm=waveletscaleaogram(saecgnorm,fs1)
    rationorm=calculate_band_power_ratio(powernorm,frequenciesnorm,300,2000)
    print(f'Доля суммарной мощности в диапазоне 300-2000 Гц при норме ={rationorm}')

    plt.figure(figsize=(15,7))
    plt.subplot(121)
    time_axis = np.arange(powernorm.shape[1]) / fs1  
    plt.imshow(powernorm, 
           extent=[time_axis[0], time_axis[-1], frequenciesnorm[-1], frequenciesnorm[0]], 
           cmap='viridis', 
           aspect='auto',
           vmax=abs(powernorm).max(), 
           vmin=-abs(powernorm).max())  
    plt.colorbar(label='Мощность')
    plt.xlabel('Время (с)')
    plt.ylabel('Частота (Гц)')
    plt.title('Вейвлет-спектрограмма')
    plt.subplot(122)
    plt.semilogy(frequenciesnorm, Spnorm)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Суммарная мощность')
    plt.title('Интегральная мощность по частотам')
    plt.grid(True)

    plt.savefig(os.path.join(save_dir_pic, "ecg_signal_averged_norm_wavelet.jpeg"), dpi=300)


    rpeaks=findpeaks(ypat,fs1)
    saecgpat=signalavergedecg(ypat,fs1,rpeaks)
    rpeakspat,qpeakspat,speakspat,qrsonpat,qrsoffpat=peaksfind(saecgpat,fs1)
    plt.figure(figsize=(15,7))
    plt.plot(np.linspace(0,len(saecgpat)/fs1,len(saecgpat)),saecgpat)
    plt.plot(np.linspace(0,len(saecgpat)/fs1,len(saecgpat))[rpeakspat],saecgpat[rpeakspat],'ro')
    plt.plot(np.linspace(0,len(saecgpat)/fs1,len(saecgpat))[speakspat],saecgpat[speakspat],'bo')
    plt.plot(np.linspace(0,len(saecgpat)/fs1,len(saecgpat))[qpeakspat],saecgpat[qpeakspat],'mo')
    plt.plot(np.linspace(0,len(saecgpat)/fs1,len(saecgpat))[qrsonpat],saecgpat[qrsonpat],'k*')
    plt.plot(np.linspace(0,len(saecgpat)/fs1,len(saecgpat))[qrsoffpat],saecgpat[qrsoffpat],'g*')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir_pic, "ecg_signal_averged_pat_peaks.jpeg"), dpi=300)

    plt.figure(figsize=(15,7))
    plt.plot(np.linspace(0,len(saecgpat)/fs1,len(saecgpat)),saecgpat)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir_pic, "ecg_signal_averged_pat.jpeg"), dpi=300)

    bin_edgespat, bin_heightspat=plot_histogram(saecgpat, save_path=os.path.join(save_dir_pic, "ecg_signal_averged_hist_pat.jpeg"))
    statsmomentspat=calculate_skew_kurtosis(saecgpat)


    total_power_pat,freq_pat, Pxx_pat=periodogram_power(saecgpat,fs1)
    bin_centerspat, bin_powerspat=plot_fixed_bin_histogram(freq_pat, Pxx_pat,save_path=os.path.join(save_dir_pic, "ecg_signal_averged_hist_psd_pat.jpeg"))
    plt.figure(figsize=(15,7))
    plt.plot(freq_pat, Pxx_pat)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir_pic, "ecg_signal_averged_psd_pat.jpeg"), dpi=300)


    Sppat,frequenciespat,powerpat=waveletscaleaogram(saecgpat,fs1)
    ratiopat=calculate_band_power_ratio(powerpat,frequenciespat,300,2000)
    print(f'Доля суммарной мощности в диапазоне 300-2000 Гц при патологии ={ratiopat}')

    plt.figure(figsize=(15,7))
    plt.subplot(121)
    time_axis = np.arange(powerpat.shape[1]) / fs1  
    plt.imshow(powernorm, 
           extent=[time_axis[0], time_axis[-1], frequenciespat[-1], frequenciespat[0]], 
           cmap='viridis', 
           aspect='auto',
           vmax=abs(powerpat).max(), 
           vmin=-abs(powerpat).max())  
    plt.colorbar(label='Мощность')
    plt.xlabel('Время (с)')
    plt.ylabel('Частота (Гц)')
    plt.title('Вейвлет-спектрограмма')
    plt.subplot(122)
    plt.semilogy(frequenciespat, Sppat)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Суммарная мощность')
    plt.title('Интегральная мощность по частотам')
    plt.grid(True)

    plt.savefig(os.path.join(save_dir_pic, "ecg_signal_averged_pat_wavelet.jpeg"), dpi=300)

    parpat=compute_late_potentials_from_avg(saecgpat,fs1)
    print(parpat)

    result=compare_signals_mannwhitney(saecgnorm,saecgpat)
    print(result)
    # Инициализируем словарь с результатами для текущего файла
    file_result = {
    'rat_number': None,  # Явно сохраняем номер крысы
    # Параметры нормального сигнала
    'norm_fQRS_ms': None,
    'norm_RMS40_mkV': None,
    'norm_LAS_ms': None,
    # Параметры патологического сигнала
    'pat_fQRS_ms': None,
    'pat_RMS40_mkV': None,
    'pat_LAS_ms': None,
    'norm_skewness':None,
    'norm_kurtosis':None,
    'norm_mean':None,
    'norm_std':None,
    'norm_QRSon':None,
    'norm_QRSoff':None,
    'norm_ratio_cwt_power':None,
    'norm_ratio_psd_power':None,
    'pat_skewness':None,
    'pat_kurtosis':None,
    'pat_mean':None,
    'pat_std':None,
    'pat_QRSon':None,
    'pat_QRSoff':None,
    'pat_ratio_cwt_power':None,
    'pat_ratio_psd_power':None,
    }
    base_path = "D:/ECG_IAI_RAS/RAT_NEW/"
    file_numbers = range(10, 19)
    results_data = []

    # Создаем прогресс-бар с дополнительной информацией
    pbar = tqdm(
    file_numbers, 
    desc="Обработка файлов", 
    unit="файл",
    ncols=100,  # ширина прогресс-бара
    colour='green',  # цвет (работает в некоторых терминалах)
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    for i in pbar:
        # Обновляем описание с текущим файлом
        pbar.set_description(f"Обработка файла {i}")
        file_result = file_result.copy()
        file_result['rat_number'] =i
        # Здесь ваш код обработки
        print(f"\n{'='*50}")
        print(f"Обработка файла {i}")
        print('='*50)

        # Формируем пути к файлам
        file_name = f"{i}_2_Not_filtered.edf"
        file_path = f"{base_path}{i}/2_rat/"
        chanel_d, time, fs1, anot = read_edf(
            file_name, 
            file_path, 
            [0, 1, 2, 3, 4, 5]
        )

        pbar.set_postfix(
            file=file_name,
            status="Обработка сигнала..."
        )
        # Дополнительная обработка сигнала
        lf_sig, hf_sig, sig1, fs1, anot = signaladd(
            file_name, 
            file_path
        )

        # Применяем винзоризацию
        sig1 = iqr_winsorize(sig1, 5)
        sig1=iircombfilter(sig1,fs1)


        # Определяем временные метки
        if not anot:
            stabilization_time = 0
            ishemia_time = 1688
            reperfusion_time = 3600
        else:
            stabilization_time = 0
            ishemia_time = next((k for k, v in anot.items() if v == 'Ishemia'), 0)
            reperfusion_time = next((k for k, v in anot.items() if v == 'Reperfusion'), 0)

        # Вырезаем участки сигнала
        ynorm = time_slice(
            sig1 - np.mean(sig1),
            stabilization_time + 5,
            stabilization_time + 34,
            fs1
        )

        ypat = time_slice(
            sig1 - np.mean(sig1),
            ishemia_time,
            ishemia_time + 34,
            fs1
        )

        pbar.set_postfix(
            file=file_name,
            status="Графики сигналов..."
        )
        # Сохраняем графики исходных сигналов
        plt.figure(figsize=(15,7))
        plt.plot(np.linspace(0, len(ynorm)/fs1, len(ynorm)), ynorm)
        plt.grid(True)
        plt.title(f'Нормальный сигнал - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_norm_2_{i}.jpeg"), dpi=300)
        plt.close()

        plt.figure(figsize=(15,7))
        plt.plot(np.linspace(0, len(ypat)/fs1, len(ypat)), ypat)
        plt.grid(True)
        plt.title(f'Патологический сигнал - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_pat_2_{i}.jpeg"), dpi=300)
        plt.close()

        pbar.set_postfix(
            file=file_name,
            status="Усреднение сигналов..."
        )

        # Обработка нормального сигнала
        rpeaks= findpeaks(ynorm, fs1)
        saecgnorm = signalavergedecg(ynorm, fs1, rpeaks)

        plt.figure(figsize=(15,7))
        plt.plot(np.linspace(0, len(saecgnorm)/fs1, len(saecgnorm)), saecgnorm)
        plt.grid(True)
        plt.title(f'Усредненный нормальный сигнал - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_norm_2_{i}.jpeg"), dpi=300)
        plt.close()
        try:
            parnorm = compute_late_potentials_from_avg(saecgnorm, fs1)
            # Сохраняем результаты
            with open(os.path.join(save_dir_txt, f"results_2_{i}_norm.txt"), 'w') as f:
                f.write(f"Результаты обработки файла {i}\n")
                f.write("="*50 + "\n")

                f.write(f"Нормальный сигнал:\n{parnorm}\n\n")
                f.close()
            # Сохраняем параметры в словарь
            if isinstance(parnorm, dict):
                file_result['norm_fQRS_ms'] = parnorm.get('fQRS_ms', None)
                file_result['norm_RMS40_mkV'] = parnorm.get('RMS40_uV', None)
                file_result['norm_LAS_ms'] = parnorm.get('LAS40_ms', None)
                file_result['norm_QRSon'] = parnorm.get('QRSon', None)
                file_result['norm_QRSoff'] = parnorm.get('QRSoff', None)
            elif isinstance(parnorm, (list, tuple)) and len(parnorm) >= 3:
                file_result['norm_fQRS_ms'] = parnorm[0]
                file_result['norm_RMS40_mkV'] = parnorm[1]
                file_result['norm_LAS_ms'] = parnorm[2]
        except:
            pass


        pbar.set_postfix(
            file=file_name,
            status="Гистограмма (норма)..."
        )
        bin_edgesnorm, bin_heightsnorm=plot_histogram(saecgnorm, save_path=os.path.join(save_dir_pic, f"ecg_signal_averged_norm_hist_2_{i}.jpeg"))
        statsmomentsnorm=calculate_skew_kurtosis(saecgnorm)
        if isinstance(statsmomentsnorm, dict):
            file_result['norm_skewness'] = statsmomentsnorm.get("skewness", None)
            file_result['norm_kurtosis'] = statsmomentsnorm.get("kurtosis", None)
            file_result['norm_std'] = statsmomentsnorm.get("std", None)
            file_result['norm_mean'] = statsmomentsnorm.get("mean", None)


        pbar.set_postfix(
            file=file_name,
            status="Спектральная плотность мощности (норма)..."
        )
        total_power_norm,freq_norm, Pxx_norm=periodogram_power(saecgnorm,fs1)
        file_result['norm_ratio_psd_power'] = total_power_norm
        plt.figure(figsize=(15,7))
        plt.plot(freq_norm, Pxx_norm)
        plt.grid(True)
        plt.title(f'Спектральная плотность мощности усредненного нормального сигнала - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_norm_psd_2_{i}.jpeg"), dpi=300)
        plt.close()
        bin_centersnorm, bin_powersnorm=plot_fixed_bin_histogram(freq_norm, Pxx_norm,save_path=os.path.join(save_dir_pic,f"ecg_signal_averged_norm_hist_psd_2_{i}.jpeg"))
        pbar.set_postfix(
            file=file_name,
            status="Вейвлет-анализ (норма)..."
        )

        # Вейвлет-анализ нормального сигнала
        Spnorm, frequenciesnorm, powernorm = waveletscaleaogram(saecgnorm, fs1)
        rationorm=calculate_band_power_ratio(powernorm,frequenciesnorm,300,2000)
        file_result['norm_ratio_cwt_power'] =rationorm
        with open(os.path.join(save_dir_txt, f"results_ratio_cwt_power_2_{i}_norm.txt"), 'w') as f:
            f.write(f"Результаты обработки файла {i}\n")
            f.write("="*50 + "\n")

            f.write(f"Нормальный сигнал:\n{rationorm}\n\n")
            f.close()

        plt.figure(figsize=(15,7))
        plt.subplot(121)
        time_axis = np.arange(powernorm.shape[1]) / fs1  
        plt.imshow(powernorm, 
               extent=[time_axis[0], time_axis[-1], frequenciesnorm[-1], frequenciesnorm[0]], 
               cmap='viridis', 
               aspect='auto',
               vmax=abs(powernorm).max(), 
               vmin=-abs(powernorm).max())  
        plt.colorbar(label='Мощность')
        plt.xlabel('Время (с)')
        plt.ylabel('Частота (Гц)')
        plt.title(f'Вейвлет-спектрограмма (норма) - Файл {i}')

        plt.subplot(122)
        plt.semilogy(frequenciesnorm, Spnorm)
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Суммарная мощность')
        plt.title('Интегральная мощность по частотам')
        plt.grid(True)

        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_norm_wavelet_2_{i}.jpeg"), dpi=300)
        plt.close()

        pbar.set_postfix(
            file=file_name,
            status="Обработка патологии..."
        )

        # Обработка патологического сигнала
        rpeaks = findpeaks(ypat, fs1)
        saecgpat = signalavergedecg(ypat, fs1, rpeaks)

        plt.figure(figsize=(15,7))
        plt.plot(np.linspace(0, len(saecgpat)/fs1, len(saecgpat)), saecgpat)
        plt.grid(True)
        plt.title(f'Усредненный патологический сигнал - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_pat_2_{i}.jpeg"), dpi=300)
        plt.close()


        pbar.set_postfix(
            file=file_name,
            status="Гистограмма (патология)..."
        )
        bin_edgespat, bin_heightspat=plot_histogram(saecgpat, save_path=os.path.join(save_dir_pic, f"ecg_signal_averged_pat_hist_2_{i}.jpeg"))
        statsmomentspat=calculate_skew_kurtosis(saecgpat)
        if isinstance(statsmomentspat, dict):
            file_result['pat_skewness'] = statsmomentspat.get("skewness", None)
            file_result['pat_kurtosis'] = statsmomentspat.get("kurtosis", None)
            file_result['pat_std'] = statsmomentspat.get("std", None)
            file_result['pat_mean'] = statsmomentspat.get("mean", None)


        pbar.set_postfix(
            file=file_name,
            status="Спектральная плотность мощности (патология)..."
        )


        total_power_pat,freq_pat, Pxx_pat=periodogram_power(saecgpat,fs1)
        file_result['pat_ratio_psd_power'] = total_power_pat
        bin_centerspat, bin_powerspat=plot_fixed_bin_histogram(freq_pat, Pxx_pat,save_path=os.path.join(save_dir_pic,f"ecg_signal_averged_pat_hist_psd_2_{i}.jpeg"))
        plt.figure(figsize=(15,7))
        plt.plot(freq_pat, Pxx_pat)
        plt.grid(True)
        plt.title(f'Спектральная плотность мощности усредненного патологического сигнала - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_pat_psd_2_{i}.jpeg"), dpi=300)
        plt.close()

        pbar.set_postfix(
            file=file_name,
            status="Вейвлет-анализ (патология)..."
        )

        # Вейвлет-анализ патологического сигнала
        Sppat, frequenciespat, powerpat = waveletscaleaogram(saecgpat, fs1)
        ratiopat=calculate_band_power_ratio(powerpat,frequenciespat,300,2000)
        file_result['pat_ratio_cwt_power'] =ratiopat
        with open(os.path.join(save_dir_txt, f"results_ratio_cwt_power_2_{i}_pat.txt"), 'w') as f:
            f.write(f"Результаты обработки файла {i}\n")
            f.write("="*50 + "\n")

            f.write(f"Нормальный сигнал:\n{ratiopat}\n\n")
            f.close()

        plt.figure(figsize=(15,7))
        plt.subplot(121)
        time_axis = np.arange(powerpat.shape[1]) / fs1  
        plt.imshow(powerpat, 
               extent=[time_axis[0], time_axis[-1], frequenciespat[-1], frequenciespat[0]], 
               cmap='viridis', 
               aspect='auto',
               vmax=abs(powerpat).max(), 
               vmin=-abs(powerpat).max())  
        plt.colorbar(label='Мощность')
        plt.xlabel('Время (с)')
        plt.ylabel('Частота (Гц)')
        plt.title(f'Вейвлет-спектрограмма (патология) - Файл {i}')

        plt.subplot(122)
        plt.semilogy(frequenciespat, Sppat)
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Суммарная мощность')
        plt.title('Интегральная мощность по частотам')
        plt.grid(True)

        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_pat_wavelet_2_{i}.jpeg"), dpi=300)
        plt.close()

        try:
           parpat = compute_late_potentials_from_avg(saecgpat, fs1)

           # Сохраняем результаты
           with open(os.path.join(save_dir_txt, f"results_2_{i}_pat.txt"), 'w') as f:
                f.write(f"Результаты обработки файла {i}\n")
                f.write("="*50 + "\n")
                f.write(f"Патологический сигнал:\n{parpat}\n")
                f.close()

           # Сохраняем параметры в словарь
           if isinstance(parpat, dict):
              file_result['pat_fQRS_ms'] = parpat.get('fQRS_ms', None)
              file_result['pat_RMS40_mkV'] = parpat.get('RMS40_uV', None)
              file_result['pat_LAS_ms'] = parpat.get('LAS40_ms', None)
              file_result['pat_QRSon'] = parpat.get('QRSon', None)
              file_result['pat_QRSoff'] = parpat.get('QRSoff', None)
           elif isinstance(parpat, (list, tuple)) and len(parpat) >= 3:
               file_result['pat_fQRS_ms'] = parpat[0]
               file_result['pat_RMS40_mkV'] = parpat[1]
               file_result['pat_LAS_ms'] = parpat[2]

        except:
            pass

        # Добавляем результаты текущего файла в общий список
        results_data.append(file_result)

    # Создаем DataFrame из собранных данных
    df_results = pd.DataFrame(results_data)





    excel_filename = os.path.join(save_dir_xlsx, "ecg_analysisresults2.xlsx")
    df_results.to_excel(excel_filename, index=False)



    file_result = {
    'rat_number': None,  # Явно сохраняем номер крысы
    # Параметры нормального сигнала
    'norm_fQRS_ms': None,
    'norm_RMS40_mkV': None,
    'norm_LAS_ms': None,
    # Параметры патологического сигнала
    'pat_fQRS_ms': None,
    'pat_RMS40_mkV': None,
    'pat_LAS_ms': None,
    'norm_skewness':None,
    'norm_kurtosis':None,
    'norm_mean':None,
    'norm_std':None,
    'norm_QRSon':None,
    'norm_QRSoff':None,
    'norm_ratio_cwt_power':None,
    'norm_ratio_psd_power':None,
    'pat_skewness':None,
    'pat_kurtosis':None,
    'pat_mean':None,
    'pat_std':None,
    'pat_QRSon':None,
    'pat_QRSoff':None,
    'pat_ratio_cwt_power':None,
    'pat_ratio_psd_power':None,
    }
    base_path = "D:/ECG_IAI_RAS/RAT_NEW/"
    file_numbers = range(10, 19)
    results_data = []

    # Создаем прогресс-бар с дополнительной информацией
    pbar = tqdm(
    file_numbers, 
    desc="Обработка файлов", 
    unit="файл",
    ncols=100,  # ширина прогресс-бара
    colour='green',  # цвет (работает в некоторых терминалах)
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    for i in pbar:
        # Обновляем описание с текущим файлом
        pbar.set_description(f"Обработка файла {i}")
        file_result = file_result.copy()
        file_result['rat_number'] =i
        # Здесь ваш код обработки
        print(f"\n{'='*50}")
        print(f"Обработка файла {i}")
        print('='*50)

        # Формируем пути к файлам
        file_name = f"{i}_2_Not_filtered.edf"
        file_path = f"{base_path}{i}/2_rat/"
        chanel_d, time, fs1, anot = read_edf(
            file_name, 
            file_path, 
            [0, 1, 2, 3, 4, 5]
        )

        pbar.set_postfix(
            file=file_name,
            status="Обработка сигнала..."
        )
        # Дополнительная обработка сигнала
        lf_sig, hf_sig, sig1, fs1, anot = signaladd(
            file_name, 
            file_path
        )

        # Применяем винзоризацию
        sig1 = iqr_winsorize(sig1, 5)
        sig1=iircombfilter(sig1,fs1)


        # Определяем временные метки
        if not anot:
            stabilization_time = 0
            ishemia_time = 1688
            reperfusion_time = 3600
        else:
            stabilization_time = 0
            ishemia_time = next((k for k, v in anot.items() if v == 'Ishemia'), 0)
            reperfusion_time = next((k for k, v in anot.items() if v == 'Reperfusion'), 0)

        # Вырезаем участки сигнала
        ynorm = time_slice(
            sig1 - np.mean(sig1),
            stabilization_time + 5,
            stabilization_time + 34,
            fs1
        )

        ypat = time_slice(
            sig1 - np.mean(sig1),
            ishemia_time,
            ishemia_time + 34,
            fs1
        )

        pbar.set_postfix(
            file=file_name,
            status="Графики сигналов..."
        )
        # Сохраняем графики исходных сигналов
        plt.figure(figsize=(15,7))
        plt.plot(np.linspace(0, len(ynorm)/fs1, len(ynorm)), ynorm)
        plt.grid(True)
        plt.title(f'Нормальный сигнал - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_norm_1_{i}.jpeg"), dpi=300)
        plt.close()

        plt.figure(figsize=(15,7))
        plt.plot(np.linspace(0, len(ypat)/fs1, len(ypat)), ypat)
        plt.grid(True)
        plt.title(f'Патологический сигнал - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_pat_1_{i}.jpeg"), dpi=300)
        plt.close()

        pbar.set_postfix(
            file=file_name,
            status="Усреднение сигналов..."
        )

        # Обработка нормального сигнала
        rpeaks = findpeaks(ynorm, fs1)
        saecgnorm = signalavergedecg(ynorm, fs1, rpeaks)

        plt.figure(figsize=(15,7))
        plt.plot(np.linspace(0, len(saecgnorm)/fs1, len(saecgnorm)), saecgnorm)
        plt.grid(True)
        plt.title(f'Усредненный нормальный сигнал - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_norm_1_{i}.jpeg"), dpi=300)
        plt.close()
        try:
            parnorm = compute_late_potentials_from_avg(saecgnorm, fs1)
            # Сохраняем результаты
            with open(os.path.join(save_dir_txt, f"results_1_{i}_norm.txt"), 'w') as f:
                f.write(f"Результаты обработки файла {i}\n")
                f.write("="*50 + "\n")

                f.write(f"Нормальный сигнал:\n{parnorm}\n\n")
                f.close()
            # Сохраняем параметры в словарь
            if isinstance(parnorm, dict):
                 file_result['norm_fQRS_ms'] = parnorm.get('fQRS_ms', None)
                 file_result['norm_RMS40_mkV'] = parnorm.get('RMS40_uV', None)
                 file_result['norm_LAS_ms'] = parnorm.get('LAS40_ms', None)
                 file_result['norm_QRSon'] = parnorm.get('QRSon', None)
                 file_result['norm_QRSoff'] = parnorm.get('QRSoff', None)
            elif isinstance(parnorm, (list, tuple)) and len(parnorm) >= 3:
                 file_result['norm_fQRS_ms'] = parnorm[0]
                 file_result['norm_RMS40_mkV'] = parnorm[1]
                 file_result['norm_LAS_ms'] = parnorm[2]
        except:
            pass



        pbar.set_postfix(
            file=file_name,
            status="Гистограмма (норма)..."
        )
        bin_edgesnorm, bin_heightsnorm=plot_histogram(saecgnorm, save_path=os.path.join(save_dir_pic, f"ecg_signal_averged_norm_hist_1_{i}.jpeg"))
        statsmomentsnorm=calculate_skew_kurtosis(saecgnorm)
        if isinstance(statsmomentsnorm, dict):
            file_result['norm_skewness'] = statsmomentsnorm.get("skewness", None)
            file_result['norm_kurtosis'] = statsmomentsnorm.get("kurtosis", None)
            file_result['norm_std'] = statsmomentsnorm.get("std", None)
            file_result['norm_mean'] = statsmomentsnorm.get("mean", None)


        pbar.set_postfix(
            file=file_name,
            status="Спектральная плотность мощности  (норма)..."
        )
        total_power_norm,freq_norm, Pxx_norm=periodogram_power(saecgnorm,fs1)
        file_result['norm_ratio_psd_power'] = total_power_norm
        plt.figure(figsize=(15,7))
        plt.plot(freq_norm, Pxx_norm)
        plt.grid(True)
        plt.title(f'Спектральная плотность мощности усредненного нормального сигнала - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_norm_psd_1_{i}.jpeg"), dpi=300)
        plt.close()

        bin_centersnorm, bin_powersnorm=plot_fixed_bin_histogram(freq_norm, Pxx_norm,save_path=os.path.join(save_dir_pic,f"ecg_signal_averged_norm_hist_psd_1_{i}.jpeg"))

        pbar.set_postfix(
            file=file_name,
            status="Вейвлет-анализ (норма)..."
        )
        # Вейвлет-анализ нормального сигнала
        Spnorm, frequenciesnorm, powernorm = waveletscaleaogram(saecgnorm, fs1)
        rationorm=calculate_band_power_ratio(powernorm,frequenciesnorm,300,2000)
        file_result['norm_ratio_cwt_power'] = rationorm
        with open(os.path.join(save_dir_txt, f"results_ratio_cwt_power_1_{i}_norm.txt"), 'w') as f:
            f.write(f"Результаты обработки файла {i}\n")
            f.write("="*50 + "\n")

            f.write(f"Нормальный сигнал:\n{rationorm}\n\n")
            f.close()

        plt.figure(figsize=(15,7))
        plt.subplot(121)
        time_axis = np.arange(powernorm.shape[1]) / fs1  
        plt.imshow(powernorm, 
               extent=[time_axis[0], time_axis[-1], frequenciesnorm[-1], frequenciesnorm[0]], 
               cmap='viridis', 
               aspect='auto',
               vmax=abs(powernorm).max(), 
               vmin=-abs(powernorm).max())  
        plt.colorbar(label='Мощность')
        plt.xlabel('Время (с)')
        plt.ylabel('Частота (Гц)')
        plt.title(f'Вейвлет-спектрограмма (норма) - Файл {i}')

        plt.subplot(122)
        plt.semilogy(frequenciesnorm, Spnorm)
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Суммарная мощность')
        plt.title('Интегральная мощность по частотам')
        plt.grid(True)

        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_norm_wavelet_1_{i}.jpeg"), dpi=300)
        plt.close()

        pbar.set_postfix(
            file=file_name,
            status="Обработка патологии..."
        )

        # Обработка патологического сигнала
        rpeaks= findpeaks(ypat, fs1)
        saecgpat = signalavergedecg(ypat, fs1, rpeaks)

        plt.figure(figsize=(15,7))
        plt.plot(np.linspace(0, len(saecgpat)/fs1, len(saecgpat)), saecgpat)
        plt.grid(True)
        plt.title(f'Усредненный патологический сигнал - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_pat_1_{i}.jpeg"), dpi=300)
        plt.close()


        pbar.set_postfix(
            file=file_name,
            status="Гистограмма (патология)..."
        )
        bin_edgesnpat, bin_heightspat=plot_histogram(saecgpat, save_path=os.path.join(save_dir_pic, f"ecg_signal_averged_pat_hist_1_{i}.jpeg"))
        statsmomentspat=calculate_skew_kurtosis(saecgpat)
        if isinstance(statsmomentspat, dict):
            file_result['pat_skewness'] = statsmomentspat.get("skewness", None)
            file_result['pat_kurtosis'] = statsmomentspat.get("kurtosis", None)
            file_result['pat_std'] = statsmomentspat.get("std", None)
            file_result['pat_mean'] = statsmomentspat.get("mean", None)


        pbar.set_postfix(
            file=file_name,
            status="Спектральная плотность мощности (патология)..."
        )
        total_power_pat,freq_pat, Pxx_pat=periodogram_power(saecgpat,fs1)
        file_result['pat_ratio_psd_power'] = total_power_pat
        plt.figure(figsize=(15,7))
        plt.plot(freq_pat, Pxx_pat)
        plt.grid(True)
        plt.title(f'Спектральная плотность мощности усредненного патологического сигнала - Файл {i}')
        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_pat_psd_1_{i}.jpeg"), dpi=300)
        plt.close()
        bin_centerspat, bin_powerspat=plot_fixed_bin_histogram(freq_pat, Pxx_pat,save_path=os.path.join(save_dir_pic, f"ecg_signal_averged_pat_hist_psd_1_{i}.jpeg"))

        pbar.set_postfix(
            file=file_name,
            status="Вейвлет-анализ (патология)..."
        )

        # Вейвлет-анализ патологического сигнала
        Sppat, frequenciespat, powerpat = waveletscaleaogram(saecgpat, fs1)
        ratiopat=calculate_band_power_ratio(powerpat,frequenciespat,300,2000)
        file_result['pat_ratio_cwt_power'] =ratiopat
        with open(os.path.join(save_dir_txt, f"results_ratio_cwt_power_1_{i}_pat.txt"), 'w') as f:
            f.write(f"Результаты обработки файла {i}\n")
            f.write("="*50 + "\n")

            f.write(f"Нормальный сигнал:\n{ratiopat}\n\n")
            f.close()

        plt.figure(figsize=(15,7))
        plt.subplot(121)
        time_axis = np.arange(powerpat.shape[1]) / fs1  
        plt.imshow(powerpat, 
               extent=[time_axis[0], time_axis[-1], frequenciespat[-1], frequenciespat[0]], 
               cmap='viridis', 
               aspect='auto',
               vmax=abs(powerpat).max(), 
               vmin=-abs(powerpat).max())  
        plt.colorbar(label='Мощность')
        plt.xlabel('Время (с)')
        plt.ylabel('Частота (Гц)')
        plt.title(f'Вейвлет-спектрограмма (патология) - Файл 1 {i}')

        plt.subplot(122)
        plt.semilogy(frequenciespat, Sppat)
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Суммарная мощность')
        plt.title('Интегральная мощность по частотам')
        plt.grid(True)

        plt.savefig(os.path.join(save_dir_pic, f"ecg_signal_averged_pat_wavelet_1_{i}.jpeg"), dpi=300)
        plt.close()

        try:
           parpat = compute_late_potentials_from_avg(saecgpat, fs1)

           # Сохраняем результаты
           with open(os.path.join(save_dir_txt, f"results_1_{i}_pat.txt"), 'w') as f:
                f.write(f"Результаты обработки файла {i}\n")
                f.write("="*50 + "\n")
                f.write(f"Патологический сигнал:\n{parpat}\n")
                f.close()
           # Сохраняем параметры в словарь
           if isinstance(parpat, dict):
              file_result['pat_fQRS_ms'] = parpat.get('fQRS_ms', None)
              file_result['pat_RMS40_mkV'] = parpat.get('RMS40_uV', None)
              file_result['pat_LAS_ms'] = parpat.get('LAS40_ms', None)
              file_result['pat_QRSon'] = parpat.get('QRSon', None)
              file_result['pat_QRSoff'] = parpat.get('QRSoff', None)
           elif isinstance(parpat, (list, tuple)) and len(parpat) >= 3:
               file_result['pat_fQRS_ms'] = parpat[0]
               file_result['pat_RMS40_mkV'] = parpat[1]
               file_result['pat_LAS_ms'] = parpat[2]

        except:
            pass

        # Добавляем результаты текущего файла в общий список
        results_data.append(file_result)

    # Создаем DataFrame из собранных данных
    df_results = pd.DataFrame(results_data)

    excel_filename = os.path.join(save_dir_xlsx, "ecg_analysisresults1.xlsx")
    df_results.to_excel(excel_filename, index=False)
    df= apply_mannwhitney_to_all(df_results)
    excel_filename = os.path.join(save_dir_xlsx, "mannwhitney1.xlsx")
    df.to_excel(excel_filename, index=False)

    X, y, feature_cols=prepare_rf_data(df_results)
    dictml=train_rf_and_plot_roc(X, y, feature_cols,save_path=os.path.join(save_dir_pic, f"rocauccurve1_{i}.jpeg"))





    plt.show()




if __name__ == "__main__":
    main()








# In[ ]:




