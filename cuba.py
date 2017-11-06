import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Wedge, Circle
import matplotlib.colors as colors
from scipy.io.wavfile import write

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

    def directedSignal(self, t, x, y):
        #gamma is angle between radial vectors of the source and the signal
        cos_gamma = (self.Rad**2+self.rprime(x,y)**2-(x**2+y**2))/(2*self.Rad*self.rprime(x,y))
        #dprime is the length along the radial vector to the source to the perpendicular intercept with the signal
        dprime = self.rprime(x,y)*cos_gamma
        sin_gamma = np.sqrt(max(0.0, 1.0-cos_gamma**2))

        modified_amplitude = self.Amp*np.exp(-(self.rprime(x,y)*sin_gamma)**2/(2*self.spread**2*dprime**2))/np.sqrt(self.rprime(x,y))
        return modified_amplitude*np.sin(self.freq*2.0*np.pi*(t-v_sound*self.rprime(x,y))+self.phi)

def generateSoundFile(soundArray, timeLength, fileName):
    scaled = np.int16(soundArray/np.max(np.abs(soundArray)) * 32767)
    write(fileName, len(scaled)/timeLength, scaled)

def plotLoudness(total_loudness, human_loudness):
    fig = plt.figure(figsize=[14.2, 6.5])
    ax = fig.add_subplot(121)
    mappable = plt.imshow(total_loudness.transpose(), extent=[x[0], x[-1], y[0], y[-1]], interpolation='none', aspect='auto',norm=colors.Normalize(vmin = 0.0, vmax = 1e7))
    c = Wedge((0., 0.), 100.0, theta1=0.0, theta2=360.0, width=500.0, edgecolor='black', facecolor='#474747')
    ax.add_patch(c)
    plt.ylim([-100.0, 100.0])
    plt.xlim([-100.0, 100.0])
    #cb = plt.colorbar()
    plt.ylabel('Y Position [m]')
    plt.xlabel('X Position [m]')
    plt.title('Total Loudness')

    ax2 = fig.add_subplot(122)
    mappable = plt.imshow(human_loudness.transpose(), extent=[x[0], x[-1], y[0], y[-1]], interpolation='none', aspect='auto')#,norm=colors.Normalize(vmin = 0.0, vmax = 1e5))

    print "Vmax = " + str(np.max(np.log10(human_loudness)))
    c2 = Wedge((0., 0.), 100.0, theta1=0.0, theta2=360.0, width=500.0, edgecolor='black', facecolor='#474747')
    ax2.add_patch(c2)
    plt.ylim([-100.0, 100.0])
    plt.xlim([-100.0, 100.0])
    #cb2 = plt.colorbar()
    plt.ylabel('Y Position [m]')
    plt.xlabel('X Position [m]')
    plt.title('Loudness below '+str(cutoff_freq/1000.0)+' KHz')
    plt.show()

total_time = 1.0e-3 #seconds
cutoff_freq = 20e3
time_res = 1e4
nyquist_freq = 2.0*time_res/total_time
tspace = np.linspace(0.0, total_time, time_res)
noise_amp = 1.0e-2

source1 = Source(Amp = 2.0, phi = np.random.rand()*2.*np.pi, freq = 100.0e3, theta = 0.0, Rad = 100.0)
source2 = Source(Amp = 2.0, phi = np.random.rand()*2.*np.pi, freq = 106.0e3, theta = 1.*2.*np.pi/8, Rad = 100.0)
source3 = Source(Amp = 2.0, phi = np.random.rand()*2.*np.pi, freq = 112.2e3, theta = 2.*2.*np.pi/8, Rad = 100.0)
source4 = Source(Amp = 2.0, phi = np.random.rand()*2.*np.pi, freq = 118.4e3, theta = 3.*2.*np.pi/8, Rad = 100.0)
source5 = Source(Amp = 2.0, phi = np.random.rand()*2.*np.pi, freq = 124.6e3, theta = 4.*2.*np.pi/8, Rad = 100.0)
source6 = Source(Amp = 2.0, phi = np.random.rand()*2.*np.pi, freq = 130.8e3, theta = 5.*2.*np.pi/8, Rad = 100.0)
source7 = Source(Amp = 2.0, phi = np.random.rand()*2.*np.pi, freq = 137.0e3, theta = 6.*2.*np.pi/8, Rad = 100.0)
source8 = Source(Amp = 2.0, phi = np.random.rand()*2.*np.pi, freq = 143.2e3, theta = 7.*2.*np.pi/8, Rad = 100.0)

sources = [source1, source2, source3, source4, source5]#, source6, source7, source8]
x = np.linspace(-100.0, 100.0, 50)
y = np.linspace(-100.0, 100.0, 50)
human_loudness = np.zeros((50, 50))
total_loudness = np.zeros((50, 50))

#Loop over position in space & calculate sound at each position
for i in range(50):
    print i
    for j in range(50):
        noise = noise_amp*(np.random.rand(int(time_res))-0.5)
        output_signal = np.zeros((len(tspace)))
        for source in sources:
            source_signal = source.directedSignal(tspace, x[i], y[j])
            output_signal += source_signal
        output_signal += noise
        #Smooth the power (i.e. the signal squared) on a time scale corresponding to the cutoff frequency
        #Not sure if this is exactly what physically happens in the ear, but it's a guess
        smoothed_signal = moving_average(output_signal**2, n=int(0.5*time_res/(total_time*cutoff_freq)))
        print int(0.5*time_res/(total_time*cutoff_freq))
        if i == 490 and j == 49:
            print "Producing sound file"
            generateSoundFile(smoothed_signal, total_time, 'output.wav')
        #Power again is proportional to the squared Fourier coefficients
        total_loudness[i,j] = np.sum((np.real((np.fft.fft(output_signal)))[1:])**2)
        human_loudness[i,j] = np.sum((np.real((np.fft.fft(smoothed_signal)))[1:int(time_res*cutoff_freq/nyquist_freq)])**2)
print "Plotting..."
plotLoudness(total_loudness, human_loudness)
