import numpy as np
from read_edf import read_edf
import pprint

from scipy.fft import fft,fftfreq,ifft
from scipy import interpolate
import json 
import yaml
from yaml.loader import SafeLoader

class parametr():
    def __init__(self,custom_config_file1=None,custom_config_file2=None):
        if custom_config_file1 is None:
            custom_config_file1="C:/Users/Admin/Downloads/paramet.yaml"
        else:
             custom_config_file1=custom_config_file1
        if custom_config_file2 is None:
            custom_config_file2="C:/Users/Admin/Downloads/paramet.json"
        else:
            custom_config_file2= custom_config_file2

    def parametrya(custom_config_file1):
        parametr_dict={}
        with open(custom_config_file1) as f:
            # читаем документ YAML
            data = yaml.load(f, Loader=SafeLoader)
            # print(type(data))
            # print(data.keys())
            parametr_dict['flffr']=data.get('flffr')
            parametr_dict['fhffr']=data.get('fhffr')
            parametr_dict['vlffr']=data.get('vlffr')
            parametr_dict['vhffr']=data.get('vhffr')
        
        #pprint.pprint(parametr_dict)
        return parametr_dict

    def parametrjs(custom_config_file2):
        parametr_dict={}
        with open(custom_config_file2) as f:
            data=json.load(f)
            #print(data.keys())
            parametr_dict['flffr']=data.get('flffr')
            parametr_dict['fhffr']=data.get('fhffr')
            parametr_dict['vlffr']=data.get('vlffr')
            parametr_dict['vhffr']=data.get('vhffr')
        #pprint.pprint(parametr_dict)
        return parametr_dict

def signaladd(file_name, folder_name,window=8192,digital_max=32768,physical_max=10,digital_min=-32767,physical_min=-10,flffr=parametr.parametrya(custom_config_file1="configfiles/parametr.yaml").get('flffr'),
              fhffr=parametr.parametrya(custom_config_file1="configfiles/parametr.yaml").get('fhffr'),vlffr=parametr.parametrya(custom_config_file1="configfiles/parametr.yaml").get('vlffr'),
            vhffr=parametr.parametrya(custom_config_file1="configfiles/parametr.yaml").get('vhffr'),k=1000,chanel_lf_name='II_LF           ',chanel_hf_name= 'II_HF           '):
    """
    file_name-path filename
    folder_name-path foldername
    k-HF gain relative to LF
    window-sliding window size
    digital_max-digital_max edf
    digital_min-digital_min edf
    physical_min-physical_min edf
    physical_max-physical_max
    chanel_lf_name-chanel lf name
    chanel hf name-chanel hf name
    flffr-lf chanel frequency amplitude frequency response from -fs/2 to fs/2
    fhffr-hf chanel frequency amplitude frequency response from -fs/2 to fs/2
    vlffr-lf chanel value amplitude frequency response
    vhffr-hf chanel value amplitude frequency response
    return raw signals lf and hf,add signal,sample rate
    """

    
    chanel_d,time,fs,anot=read_edf(file_name, folder_name,[0, 1, 2, 3, 4, 5],read_annotation=True)
    pprint.pprint(chanel_d)
    print('anot')
    pprint.pprint(anot)
    print()
    
    
    sig1=chanel_d.get(chanel_lf_name)
    sig2=chanel_d.get(chanel_hf_name)
    sig20=sig2
    fs1=fs
    # print(sig1.dtype)
    # print(sig2.dtype)
    sig1=np.asarray(sig1,dtype=np.float64)
    sig2=np.asarray(sig2,dtype=np.float64)
    y = np.zeros_like(sig1,dtype=np.float64)
    # print(sig1.dtype)
    # print(sig2.dtype)
    units_per_bit = (physical_max - physical_min) / (digital_max - digital_min)
    offset = (physical_max / units_per_bit) - digital_max
    # print(offset)
    
    
    sig1=sig1*units_per_bit
    sig2=sig2*units_per_bit/k
    if len(sig1)==len(sig2):
        N=len(sig1)
        n=0
        
        
        while n < N - window:
            y1 = np.array(sig1[n:n + window])
            y2 = np.array(sig2[n:n + window])
            half=len(y1)//2
            f=fftfreq(len(y1), 1/fs1)
            ywf1=fft(y1)
            ywf2=fft(y2)
            x1=f[:half]
            z1=np.abs(ywf1[:half])
            z2=np.abs(ywf2[:half])
            
            
            f1 = interpolate.interp1d(flffr, vlffr)
            afchlf = f1(f)
            afchlf=(afchlf-np.amin(afchlf))/np.ptp(afchlf)
            #print(np.amax(afchlf),np.amin(afchlf))
            f2 = interpolate.interp1d(fhffr, vhffr)
            afchhf = f2(f)
            afchhf=(afchhf-np.amin(afchhf))/np.ptp(afchhf)
            #print(np.amax(afchhf),np.amin(afchhf))
            
            ywf1=afchlf/(afchlf+ afchhf)*ywf1
            ywf2=afchhf/(afchlf+ afchhf)*ywf2
            ywf=np.add(ywf1,ywf2)
            y[n:n + window]=np.real(ifft(ywf))
            n += window
        
        
        # print(y.shape)
        # print(sig1.shape)
        print(f'max={np.amax(y)}')

        
        

    return sig1,sig20,y,fs,anot


def main():
    help(signaladd)
if __name__ == "__main__":
    main()