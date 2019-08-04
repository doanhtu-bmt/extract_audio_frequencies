'''
Created on Jul 28, 2019

@author: discus
'''

from numpy import array, diff, where, split
from scipy import arange
import soundfile
import numpy, scipy
import pylab
import copy
import matplotlib
matplotlib.use('tkagg')

def findPeak(magnitude_values, noise_level=2000):
    
    splitter = 0
    # zero out low values in the magnitude array to remove noise (if any)
    magnitude_values = numpy.asarray(magnitude_values)        
    low_values_indices = magnitude_values < noise_level  # Where values are low
    magnitude_values[low_values_indices] = 0  # All low values will be zero out
    
    indices = []
    
    flag_start_looking = False
    
    both_ends_indices = []
    
    length = len(magnitude_values)
    for i in range(length):
        if magnitude_values[i] != splitter:
            if not flag_start_looking:
                flag_start_looking = True
                both_ends_indices = [0, 0]
                both_ends_indices[0] = i
        else:
            if flag_start_looking:
                flag_start_looking = False
                both_ends_indices[1] = i
                # add both_ends_indices in to indices
                indices.append(both_ends_indices)
                
    return indices

def extractFrequency(indices, freq_threshold=2):
    
    extracted_freqs = []
    
    for index in indices:
        freqs_range = freq_bins[index[0]: index[1]]
        avg_freq = round(numpy.average(freqs_range))
        
        if avg_freq not in extracted_freqs:
            extracted_freqs.append(avg_freq)

    # group extracted frequency by nearby=freq_threshold (tolerate gaps=freq_threshold)
    group_similar_values = split(extracted_freqs, where(diff(extracted_freqs) > freq_threshold)[0]+1 )
    
    # calculate the average of similar value
    extracted_freqs = []
    for group in group_similar_values:
        extracted_freqs.append(round(numpy.average(group)))
    
    print("freq_components", extracted_freqs)
    return extracted_freqs

if __name__ == '__main__':
    
    file_path = 'sin_1000Hz_-3dBFS_10s.wav'
    print('Open audio file path:', file_path)
    
    audio_samples, sample_rate  = soundfile.read(file_path, dtype='int16')
    number_samples = len(audio_samples)
    print('Audio Samples: ', audio_samples)
    print('Number of Sample', number_samples)
    print('Sample Rate: ', sample_rate)
    
    # duration of the audio file
    duration = round(number_samples/sample_rate, 2)
    print('Audio Duration: {0}s'.format(duration))
    
    # list of possible frequencies bins
    freq_bins = arange(number_samples) * sample_rate/number_samples
    print('Frequency Length: ', len(freq_bins))
    print('Frequency bins: ', freq_bins)
    
#     # FFT calculation
    fft_data = scipy.fft(audio_samples)
    print('FFT Length: ', len(fft_data))
    print('FFT data: ', fft_data)

    freq_bins = freq_bins[range(number_samples//2)]      
    normalization_data = fft_data/number_samples
    magnitude_values = normalization_data[range(len(fft_data)//2)]
    magnitude_values = numpy.abs(magnitude_values)
        
    indices = findPeak(magnitude_values=magnitude_values, noise_level=200)
    frequencies = extractFrequency(indices=indices)
    print("frequencies:", frequencies)
    
    x_asis_data = freq_bins
    y_asis_data = magnitude_values
 
    pylab.plot(x_asis_data, y_asis_data, color='blue') # plotting the spectrum
  
    pylab.xlabel('Freq (Hz)')
    pylab.ylabel('|Magnitude - Voltage  Gain / Loss|')
    pylab.show()
