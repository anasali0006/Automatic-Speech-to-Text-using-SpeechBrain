import speechbrain
from speechbrain.pretrained import EncoderDecoderASR
import torch
import torchaudio
import numpy as np
import time
from jiwer import wer
import json
import os

############# select a model  ################### 
asr_model3 = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech")


os.chdir('dump')
print(os.getcwd)
########## original transcripts dictionary #############
dictionary_original={}

with open('../LibriSpeechWave/new_transcriptions.txt', 'r', encoding='utf-8-sig') as file:
  for i in file:
    list1=i.split('\t')
    dictionary_original[list1[0]]=list1[1].strip('\n')

######### results dictionary ###############
result_dict={}
#to contain data as 
# key: file_name
# values: (wer, original_text, model_output_text, audio_time, model_inference_time)
for i in dictionary_original.keys():  
  path='../LibriSpeechWave/'+ i 
  #print(path)
  audio, fs = torchaudio.load(path)
  audio_time=np.shape(audio)[1] / fs
  # print(asr_model3.transcribe_file(path))
  t1=time.time()
  result=asr_model3.transcribe_file(path)
  t2=time.time()
  inference_time=t2-t1
  wer_value= wer (dictionary_original[i], result)
  result_dict[i]=(wer_value, dictionary_original[i], result, audio_time, inference_time )
  print('Progress:{}/{}'.format(len(result_dict), len(dictionary_original)))

os.chdir('..')

json_object=json.dumps(result_dict, indent = 4)
with open("results.json", "w") as outfile:
    outfile.write(json_object)