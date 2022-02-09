from jiwer import wer
import json
import matplotlib.pyplot as plt
import numpy as np

# json_file = open('results.json')
json_file = open('Results_colab.json')
loaded_dictionary = json.load(json_file)

#creating a list of WER, audio_length, inference_time
WER_list=[]
audio_time_list=[]
inference_time_list=[]
for i in loaded_dictionary.values():
  WER_list.append(i[0])
  audio_time_list.append(i[3])
  inference_time_list.append(i[4])

# Average of all WERs
weighted_wer_list=[]
for i in loaded_dictionary.values():
    wer_temp=i[0]
    weighted_wer = wer_temp * len(i[1].split())
    weighted_wer_list.append(weighted_wer)

# print('Average of WERs:',np.average(WER_list))

# Overall WERs
original_text_all=''
result_text_all=''

for i in loaded_dictionary.values():
  original_text_all=original_text_all + i[1]
  result_text_all=result_text_all + i[2]

print('Weighted Average:', np.sum(weighted_wer_list)/len(original_text_all.split()))
print('Overall WER:',wer(original_text_all, result_text_all))


# Word Error Rate frequency distribution
# plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
# plt.hist(WER_list, bins=100)
# plt.gca().set(title='Frequency Histogram', ylabel='Frequency', xlabel='WER');
# plt.savefig('WER.jpg')

#Inference time vs Audio Time
print('Total Audio Time:', np.sum(audio_time_list)/60)
print('Total Inference Time:', np.sum(inference_time_list)/60)