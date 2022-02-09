from email.mime import audio
import speechbrain
from speechbrain.pretrained import EncoderDecoderASR
import torch
import torchaudio
import numpy as np
import time
from jiwer import wer
import json
import os
import matplotlib.pyplot as plt

############# select a model  ################### 
asr_model3 = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech")

#path='367-130732-0000.wav'
path = '6128-63240-0008.wav'

# result=asr_model3.transcribe_file(path)
# print(result)

# Read Audio 
signal, sr = torchaudio.load(str(path), channels_first=False)
audio_time=np.shape(signal)[0] / sr



#dividing the audio in 15 seconds chunks
energy = np.square (signal)
chunk_size_15 = sr * 15 
counter = 0 
audio_chunks=[]
front=0
back=0

for idx, val in enumerate (signal):
    counter += 1
    if counter >= chunk_size_15 and energy[idx] < 0.01:
        front = idx
        audio_chunks.append(signal[back:front])
        back = front + 1
        counter = 0 
    if idx == len(signal) - 1 and counter!=0 :
        if counter > chunk_size_15/2:
            audio_chunks.append(signal[back:idx])
        else:
            audio_chunks[-1] = torch.cat((audio_chunks[-1],signal[back:idx]))

#######



# #visualize the audio
# plt.figure(1)
# plt.title("Audio")
# plt.plot(energy)
# plt.savefig("audio.png")
# print(energy.max())

for i in audio_chunks:


    # Perfrom Normalization
    normalized = asr_model3.audio_normalizer(i, sr)

    # fake a batch
    batch = normalized.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    predicted_words, predicted_tokens = asr_model3.transcribe_batch(
        batch, rel_length
    )

    print(predicted_words[0])


 



