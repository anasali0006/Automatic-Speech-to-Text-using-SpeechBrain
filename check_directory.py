import os
import torch
import torchaudio
import numpy as np
import time
import dask.array as da

# dictionary_original={}

# os.chdir('dump')
# print(os.getcwd())


# with open('../LibriSpeechWave/new_transcriptions.txt', 'r', encoding='utf-8-sig') as file:
#   for i in file:
#     list1=i.split('\t')
#     dictionary_original[list1[0]]=list1[1].strip('\n')


# os.chdir('..')
# print(os.getcwd())

# print(dictionary_original)

# audio, fs = torchaudio.load('LibriSpeechWave/367-130732-0001.wav')
# audio_time=np.shape(audio)[1] / float(fs)

# print(audio_time)

t1=time.time()
array1=da.ones((10000,10000))
array2=da.ones((10000,10000))
mul=da.matmul(array1,array2)
t2=time.time()

print(t2-t1)

array1=np.ones((1000,1000))
array2=np.ones((1000,1000))

mul=np.matmul(array1,array2)



t3=time.time()
print(t3-t2)
