import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Wedge, Circle
import matplotlib.colors as colors

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

v_sound = 340 #m/s

class Source():
    def __init__(self, Amp, phi, freq, theta, Rad, spread = np.pi/20):
        self.Amp = Amp
        self.phi = phi
        self.freq = freq
        self.theta = theta
        self.Rad = Rad
        self.spread = spread
        return None


    def rprime(self, x, y):
        return np.sqrt((self.Rad*np.cos(self.theta)-x)**2+(self.Rad*np.sin(self.theta)-y)**2)

    def signal(self, t, x, y):
        return self.Amp/self.rprime(x,y)*np.sin(self.freq*2.0*np.pi*(t-v_sound*self.rprime(x,y))+self.phi)

    def directed_signal(self, t, x, y):
        #gamma is angle between radial vectors of the source and the signal
        cos_gamma = (self.Rad**2+self.rprime(x,y)**2-(x**2+y**2))/(2*self.Rad*self.rprime(x,y))
        #dprime is the length along the radial vector to the source to the perpendicular intercept with the signal
        dprime = self.rprime(x,y)*cos_gamma
        sin_gamma = np.sqrt(1.0-cos_gamma**2)
        modified_amplitude = self.Amp*np.exp(-(self.rprime(x,y)*sin_gamma)**2/(2*self.spread**2*dprime**2))/np.sqrt(self.rprime(x,y))
        return modified_amplitude*np.sin(self.freq*2.0*np.pi*(t-v_sound*self.rprime(x,y))+self.phi)

total_time = 1e-3 #seconds
time_res = 1.0e4
cutoff_freq = 20e3
nyquist_freq = 2.0*time_res/total_time
tspace = np.linspace(0.0, total_time, time_res)



source1 = Source(Amp = 1.2, phi = 0.0, freq = 100e3, theta = 0.0, Rad = 100.0)
source2 = Source(Amp = 1.2, phi = np.pi/2, freq = 108e3, theta = np.pi/5, Rad = 100.0)
source3 = Source(Amp = 1.2, phi = 1.2*np.pi/5, freq = 112e3, theta = 2*np.pi/5, Rad = 100.0)
source4 = Source(Amp = 1.2, phi = 2.0*np.pi/5, freq = 104e3, theta = 3.*np.pi/5, Rad = 100.0)
source5 = Source(Amp = 1.2, phi = 4.0*np.pi/5, freq = 106e3, theta = 4.*np.pi/5, Rad = 100.0)

sources = [source1, source2, source3, source4, source5]
x = np.linspace(-100.0, 100.0, 50)
y = np.linspace(-100.0, 100.0, 50)
human_loudness = np.zeros((50, 50))
total_loudness = np.zeros((50, 50))

for i in range(50):
    print i
    for j in range(50):
        noise_amp = 0.0#5.0e-5
        noise = 0.#noise_amp*(np.random.rand(int(time_res))-0.5)
        output_signal = np.zeros((len(tspace)))
        for source in sources:
            output_signal += source.directed_signal(tspace, x[i], y[j])
        smoothed_signal = moving_average(output_signal**2, n=int(0.5*time_res/(total_time*cutoff_freq)))
        if i == 25 and j == 235:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            plt.plot(tspace[:len(smoothed_signal)], smoothed_signal, color='red')

            plt.title('Sound')
            ax2 = fig.add_subplot(212)
            plt.plot(np.fft.fft(smoothed_signal)[1:int(time_res/2)]**2)
            plt.title('Fourier Transform')
            plt.xscale('log')
            plt.show()

        total_loudness[i,j] = np.sum((np.real((np.fft.fft(output_signal)))[1:])**2)
        human_loudness[i,j] = np.sum((np.real((np.fft.fft(smoothed_signal)))[1:int(time_res*cutoff_freq/nyquist_freq)])**2)


fig = plt.figure(figsize=[14.2, 5.5])
ax = fig.add_subplot(121)
mappable = plt.imshow(total_loudness.transpose(), extent=[x[0], x[-1], y[0], y[-1]], interpolation='none', aspect='auto',norm=colors.Normalize(vmin = 0.0, vmax = np.max(total_loudness)))
c = Wedge((0., 0.), 100.0, theta1=0.0, theta2=360.0, width=500.0, edgecolor='black', facecolor='#474747')
ax.add_patch(c)
plt.ylim([-100.0, 100.0])
plt.xlim([-100.0, 100.0])
cb = plt.colorbar()
plt.ylabel('Y Position [m]')
plt.xlabel('X Position [m]')
plt.title('Total Loudness')

ax2 = fig.add_subplot(122)
mappable = plt.imshow(human_loudness.transpose(), extent=[x[0], x[-1], y[0], y[-1]], interpolation='none', aspect='auto',norm=colors.Normalize(vmin = 0.0, vmax = np.max(human_loudness)))
c2 = Wedge((0., 0.), 100.0, theta1=0.0, theta2=360.0, width=500.0, edgecolor='black', facecolor='#474747')
ax2.add_patch(c2)
plt.ylim([-100.0, 100.0])
plt.xlim([-100.0, 100.0])
cb2 = plt.colorbar()
plt.ylabel('Y Position [m]')
plt.xlabel('X Position [m]')
plt.title('Loudness below '+str(cutoff_freq/1000.0)+' KHz')
plt.show()


total_time = 1e-3 #seconds
time_res = 1.0e4
cutoff_freq = 20e3
tspace = np.linspace(0.0, total_time, time_res)

noise_amp = 5.0e-4
noise = noise_amp*(np.random.rand(int(time_res))-0.5)
output_signal = noise+source1.signal(tspace, 0.0, 0.0)+source2.signal(tspace, 0.0, 0.0)
