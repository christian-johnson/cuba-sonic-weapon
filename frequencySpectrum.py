import numpy as np
import matplotlib.pyplot as plt

file = open('cubaRecording.wav','rb')
data = np.fromfile(file,dtype=np.uint8)[10000:]
l = int(len(data)/10.)


for i in range(10):
	g = data[i*l:(i+1)*l]
	x = np.fft.fft(g)
	y = np.sqrt(np.real(x)**2+np.imag(x)**2)
	fig = plt.figure(figsize=[8,8])
	plt.plot(y[2500:5000])
	plt.ylim([0, 200000])
	plt.savefig('plots/0'+str(i)+'.png',bbox_inches='tight')
	plt.close()
