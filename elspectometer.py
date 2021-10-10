import os
import librosa
import librosa.display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys


# READ IN AUDIO FILES 


path_in='customer'
files_sub=os.listdir(path_in)
for file in files_sub:
        
#         print (numbers[6],emotion)
    path_save = 'customer/logmel_data/{0}.jpeg'.format(file.split('.')[0])
    path_load='customer/{0}'.format(file)
    path_save_dir = 'customer/logmel_data/'
    if not os.path.isdir(path_save_dir):os.mkdir(path_save_dir)
    y, sr = librosa.load(path_load)
    yt,_=librosa.effects.trim(y)
    y=yt
#         y= stretch(y)
    y = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
    db_spec = librosa.power_to_db(y)
    librosa.display.specshow(db_spec, y_axis='mel', fmax=20000, x_axis='time');
    plt.savefig(path_save)
    print(file,' done!')