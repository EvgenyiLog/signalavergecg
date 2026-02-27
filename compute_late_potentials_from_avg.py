import numpy as np
from typing import Dict, Tuple, Optional
from scipy.signal import butter, filtfilt


def bandpass_saecg(
    x: np.ndarray,
    fs: float,
    low: float = 40.0,
    high: float = 250.0,
    order: int = 2
) -> np.ndarray:
    """
    Полосовая фильтрация SAECG сигнала.

    Parameters
    ----------
    x : np.ndarray
        Усреднённый кардиоцикл (желательно в мкВ).
    fs : float
        Частота дискретизации, Гц.
    low : float, optional
        Нижняя граница полосы (по умолчанию 40 Гц).
    high : float, optional
        Верхняя граница полосы (по умолчанию 250 Гц).
    order : int, optional
        Порядок фильтра Баттерворта.

    Returns
    -------
    np.ndarray
        Отфильтрованный сигнал.

    Notes
    -----
    Фильтрация 40–250 Гц соответствует классике SAECG
    для анализа поздних потенциалов.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def detect_qrs_bounds(
    x_f: np.ndarray,
    fs: float,
    noise_portion: float = 0.2,
    k: float = 4.0
) -> Tuple[int, int]:
    """
    Автоматическое определение границ QRS в SAECG.

    Parameters
    ----------
    x_f : np.ndarray
        Высокочастотно фильтрованный сигнал.
    fs : float
        Частота дискретизации, Гц.
    noise_portion : float, optional
        Доля начального участка для оценки шума.
    k : float, optional
        Множитель порога (обычно 3–5).

    Returns
    -------
    (qrs_on, qrs_off) : Tuple[int, int]
        Индексы начала и конца QRS.

    Raises
    ------
    ValueError
        Если QRS не обнаружен.

    Notes
    -----
    Используется огибающая |x_f| и адаптивный порог:

        threshold = k * std(noise)

    Метод устойчив для усреднённых комплексов.
    """
    y = np.abs(x_f)

    n_noise = max(10, int(len(x_f) * noise_portion))
    noise_std = np.std(y[:n_noise])
    thr = k * noise_std

    idx = np.where(y > thr)[0]
    if len(idx) == 0:
        raise ValueError("QRS не обнаружен — проверьте сигнал.")

    return int(idx[0]), int(idx[-1])


def compute_late_potentials_from_avg(
    x: np.ndarray,
    fs: float,
    las_threshold_uv: float = 40.0,
    return_filtered: bool = False
) -> Dict[str, float]:
    """
    Расчёт fQRS, RMS40 и LAS40 по усреднённому кардиоциклу.

    Parameters
    ----------
    x : np.ndarray
        Усреднённый кардиоцикл (SAECG).
        ⚠️ Желательно в микровольтах.
    fs : float
        Частота дискретизации, Гц.
    las_threshold_uv : float, optional
        Порог для LAS40 (по умолчанию 40 мкВ).
    return_filtered : bool, optional
        Вернуть ли отфильтрованный сигнал.

    Returns
    -------
    Dict[str, float]
        Словарь с параметрами:

        - fQRS_ms
        - RMS40_uV
        - LAS40_ms
        - QRSon
        - QRSoff
        - noise_std

    Notes
    -----
    Алгоритм:

    1. Полосовая фильтрация 40–250 Гц
    2. Оценка шума
    3. Автодетекция QRS
    4. Расчёт метрик:

       fQRS — длительность фильтрованного QRS  
       RMS40 — СКЗ последних 40 мс  
       LAS40 — длительность < 40 мкВ перед QRSoff

    Clinical thresholds (классика):

    - fQRS > 114–120 мс
    - RMS40 < 20 мкВ
    - LAS40 > 38–40 мс

    Late potentials считаются положительными,
    если ≥ 2 критерия выполнены.
    """

    # ---------- фильтрация ----------
    x_f = bandpass_saecg(x, fs)

    # ---------- границы QRS ----------
    qrs_on, qrs_off = detect_qrs_bounds(x_f, fs)

    # ---------- fQRS ----------
    fQRS_ms = (qrs_off - qrs_on) / fs * 1000.0

    # ---------- RMS40 ----------
    n40 = max(1, int(round(0.040 * fs)))
    start = max(0, qrs_off - n40 + 1)
    seg = x_f[start:qrs_off + 1]
    RMS40_uV = float(np.sqrt(np.mean(seg ** 2)))

    # ---------- LAS40 ----------
    y = np.abs(x_f)
    count = 0
    for i in range(qrs_off, -1, -1):
        if y[i] < las_threshold_uv:
            count += 1
        else:
            break
    LAS40_ms = count / fs * 1000.0

    # ---------- шум ----------
    noise_std = float(np.std(np.abs(x_f[: max(10, int(0.2 * len(x_f)))])))

    result = {
        "fQRS_ms": float(fQRS_ms),
        "RMS40_uV": float(RMS40_uV),
        "LAS40_ms": float(LAS40_ms),
        "QRSon": int(qrs_on),
        "QRSoff": int(qrs_off),
        "noise_std": noise_std,
    }

    if return_filtered:
        result["x_filtered"] = x_f

    return result