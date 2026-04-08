import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, sosfiltfilt

def remove_baseline_drift(
    signal: ArrayLike,
    fs: float,
    cutoff: float = 0.7,
    order: int = 2
) -> NDArray[np.float64]:
    """Удаляет низкочастотный дрейф базовой линии фильтром Баттерворта ВЧ.

    Применяет фильтр нулевой фазы (`sosfiltfilt`), что сохраняет морфологию
    и временные метки исходного сигнала. Стандартный выбор для ЭКГ, ЭЭГ,
    акселерометрии и других биомедицинских/инженерных сигналов.

    Args:
        signal: Входной одномерный сигнал (список, кортеж или np.ndarray).
        fs: Частота дискретизации в Гц.
        cutoff: Частота среза фильтра в Гц. По умолчанию 0.7 Гц.
        order: Порядок фильтра Баттерворта (2–4 обычно достаточно).
               Фактический порядок после двойного прохода `sosfiltfilt` равен `2 * order`.

    Returns:
        Отфильтрованный сигнал в виде 1D `numpy.ndarray` с dtype `float64`.

    Raises:
        ValueError: Если сигнал не одномерный, частота среза ≥ частоты Найквиста,
                    порядок ≤ 0 или длина сигнала недостаточна для стабильного
                    применения фильтра.

    Example:
        >>> import numpy as np
        >>> fs = 1000  # Гц
        >>> t = np.linspace(0, 5, fs * 5, endpoint=False)
        >>> raw = np.sin(2 * np.pi * 1.5 * t) + 2.0 * np.sin(2 * np.pi * 0.2 * t)  # сигнал + дрейф
        >>> clean = remove_baseline_drift(raw, fs=fs, cutoff=0.7)
    """
    sig = np.asarray(signal, dtype=np.float64)

    if sig.ndim != 1:
        raise ValueError("Сигнал должен быть одномерным (1D).")
    if cutoff >= fs / 2.0:
        raise ValueError(f"Частота среза ({cutoff} Гц) должна быть строго меньше частоты Найквиста ({fs/2:.1f} Гц).")
    if order <= 0:
        raise ValueError("Порядок фильтра должен быть положительным целым числом.")

    # sosfiltfilt автоматически дополняет края, но требует минимальной длины сигнала
    min_length = 3 * (2 * order + 1)
    if sig.size < min_length:
        raise ValueError(
            f"Длина сигнала ({sig.size}) слишком мала. Для порядка {order} требуется минимум {min_length} отсчётов."
        )

    # SOS-формат гарантирует численную стабильность даже при высоких порядках
    sos = butter(order, cutoff, btype="high", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)