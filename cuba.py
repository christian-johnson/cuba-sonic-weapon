import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

v_sound = 340 #m/s

class Source():
    def __init__(self, Amp, phi, freq, theta, Rad):
        self.Amp = Amp
        self.phi = phi
        self.freq = freq
        self.theta = theta
        self.Rad = Rad
        return None
        
    def rprime(self, x, y):
        return np.sqrt((self.Rad*np.cos(self.theta)-x)**2+(self.Rad*np.sin(self.theta)-y)**2)

    def signal(self, t, x, y):
        return self.Amp/self.rprime(x,y)**2*np.sin(self.freq*2.0*np.pi*(t-v_sound*self.rprime(x,y))+self.phi)
        

total_time = 1e-3 #seconds
time_res = 1.0e4
cutoff_freq = 20e3
tspace = np.linspace(0.0, total_time, time_res)


    
source1 = Source(Amp = 1.0, phi = 0.0, freq = 100e3, theta = 0.0, Rad = 100.0)
source2 = Source(Amp = 0.75, phi = np.pi/5, freq = 108e3, theta = np.pi/3, Rad = 100.0)
source3 = Source(Amp = 1.0, phi = 0.0, freq = 118, theta = 2.0*np.pi/3, Rad = 100.0)
source4 = Source(Amp = 0.75, phi = np.pi/5, freq = 114e3, theta = 3.0*np.pi/3, Rad = 100.0)
source5 = Source(Amp = 1.0, phi = 0.0, freq = 112e3, theta = 4.0*np.pi/3, Rad = 100.0)
source6 = Source(Amp = 0.75, phi = np.pi/5, freq = 110e3, theta = 5.0*np.pi/3, Rad = 100.0)

sources = [source1, source2, source3, source4, source5, source6]
x = np.linspace(-50.0, 50.0, 50)
y = np.linspace(-50.0, 50.0, 50)

loudness = np.zeros((50, 50))
for i in range(50):
    print i
    for j in range(50):
        noise_amp = 5.0e-5
        noise = noise_amp*(np.random.rand(int(time_res))-0.5)
        output_signal = np.zeros((len(tspace)))
        for source in sources:
            output_signal += source.signal(tspace, x[i], y[j])
        loudness[i, j] = np.sum(np.real(np.fft.fft(moving_average(output_signal**2, n=int(0.5*time_res/(total_time*cutoff_freq))))[1:int(0.5*time_res/(total_time*cutoff_freq))])**2)
plt.imshow(loudness)
plt.show()
    

total_time = 1e-3 #seconds
time_res = 1.0e4
cutoff_freq = 20e3
tspace = np.linspace(0.0, total_time, time_res)

noise_amp = 5.0e-4
noise = noise_amp*(np.random.rand(int(time_res))-0.5)
output_signal = noise+source1.signal(tspace, 0.0, 0.0)+source2.signal(tspace, 0.0, 0.0)

fig = plt.figure()
ax1 = fig.add_subplot(211)
avg_signal = moving_average(output_signal**2, n=int(0.5*time_res/(total_time*cutoff_freq)))
plt.plot(tspace[:len(avg_signal)], avg_signal, color='red')

plt.title('Sound')
ax2 = fig.add_subplot(212)
plt.plot(np.fft.fft(avg_signal)[1:int(time_res/2)]**2)
plt.title('Fourier Transform')
plt.xscale('log')
plt.show()
