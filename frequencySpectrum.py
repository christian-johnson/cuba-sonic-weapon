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
min_freq = 6500
max_freq = 8000.0

fig = plt.figure(figsize=[8,8])
plt.imshow((heatMap[np.argmin(np.abs(freq-min_freq)):np.argmin(np.abs(freq-max_freq)),:])**(0.5),aspect='auto',extent=[0, totalTime, max_freq, min_freq], cmap='viridis')

def dopplerShift(fe, c, v):
	return fe*(1.+v/c)

def v(w, r, t, c, phi):
	return w*r*np.cos(w*t+phi)

def loudness(w, t, phi): #The emitter is blocked as it goes around pi->2*pi. Observer can hear it from 0->pi
	return 1.0-np.heaviside((w*t+phi)%(2.*np.pi)-np.pi, 0.0)

#parameters
T = 5.0 #s
w = 2.*np.pi/T
r = 0.98 # m
gap = 1.0 #m
c = 346.43 #m/s
fe = 7210.0
phi_0 = np.pi/3.0
fund_freq = c/(2.*r) #if the middle of the tube is open
t = np.linspace(0.0, totalTime, 1000.0)
for i in range(9):
	feff = fe+(i-4)*fund_freq
	fO = loudness(w, t, phi=0.0+phi_0)*dopplerShift(fe=feff, c=c, v=v(w, r+gap/2.0, t, c, phi=0.0+phi_0))+loudness(w, t, phi=np.pi+phi_0)*dopplerShift(fe=feff, c=c, v=v(w, r+gap/2.0, t, c, phi=np.pi+phi_0))
	plt.plot(t, fO, lw = 1.0-0.2*np.abs(i-4), ls = '--', c='white')

plt.ylim([max_freq, min_freq])
plt.gca().invert_yaxis()
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.show()
