from EDFlib.edfreader import EDFreader
import numpy as np
import scipy.io.wavfile


def create_time(data, samplerate):
    """Create time for signal"""
    length = len(data) / samplerate
    return np.linspace(0., length, len(data))



def read_edf(file_name, folder_name='', channels=[0], start_sample=0, finish_sample=None, read_annotation=None):
    """Read edf-file from one channel and return data, time and samplerate"""
    file = EDFreader(folder_name + file_name)
    samplerate = file.getSampleFrequency(channels[0])

    len_signal = file.getTotalSamples(channels[0])
    finish_sample = len_signal if finish_sample is None else finish_sample

    channel_dict = {}
    for channel in channels:
        data = np.empty(len_signal, dtype=np.int16)
        file.readSamples(channels[channel], data, finish_sample)
        channel_dict[file.getSignalLabel(channel)] = data[start_sample:finish_sample]

    if read_annotation:
        read_annotation = {}
        for annot in file.annotationslist:
            read_annotation[annot[0] / 10000000] = annot[2]

    time = create_time(data, samplerate)
    return channel_dict, time[start_sample:finish_sample], samplerate, read_annotation