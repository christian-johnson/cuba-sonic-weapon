import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io.wavfile import read

sampleRate, rawData = read('cubaRecording.wav','rb')
nFrames = rawData.shape[0]
wavData = np.array(rawData, dtype='float')*1.0/256.
print(nFrames)
totalTime = nFrames/sampleRate
print('Total time = ' +str(totalTime)+' s')
nTimeBins = 100
print('Frequency of samples: ' + str(1.0/(totalTime/nTimeBins))+' Hz')
timeBinLength = totalTime/nTimeBins
windowLength = int(nFrames/nTimeBins)

heatMap = np.zeros((len(np.fft.rfft(wavData[:windowLength,0])), nTimeBins))
for i in range(nTimeBins):
	dataLeft = wavData[i*windowLength:(i+1)*windowLength, 0]
	dataRight = wavData[i*windowLength:(i+1)*windowLength, 1]
	fftLeft = np.fft.rfft(dataLeft)
	fftRight = np.fft.rfft(dataRight)
	powerLeft = np.sqrt(np.real(fftLeft)**2+np.imag(fftLeft)**2)
	powerRight = np.sqrt(np.real(fftRight)**2+np.imag(fftRight)**2)
	heatMap[:,i] = np.mean([powerLeft, powerRight], axis=0)

freq = np.fft.rfftfreq(dataLeft.shape[-1],d=1./sampleRate)
min_freq = 6000
max_freq = 8500

fig = plt.figure(figsize=[8,8])
plt.imshow((heatMap[np.argmin(np.abs(freq-min_freq)):np.argmin(np.abs(freq-max_freq)),:])**(0.5),aspect='auto',extent=[0, totalTime, max_freq, min_freq], cmap='viridis')
plt.gca().invert_yaxis()
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.show()
