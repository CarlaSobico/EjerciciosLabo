import numpy as np
import math as m
import scipy as sci
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftshift
import pylab as pl

k1 = 0.19 * 10e12
k2 = -1.9 * 10e6
fi0 = 0

fs = 50e6

#x=np.arange(N) / float(fs)
x=np.r_[0:(10e-6-1/fs):1/(fs)]
print(len(x))

x2=np.r_[0:(50e-6):1/(fs)]
print(len(x2))

x3=np.r_[0:(60e-6):1/(fs)]
print(len(x3))

x4=np.r_[0:(40e-6):1/(fs)]
print(len(x2))

#x=np.linspace(0,10e-6,M)

def theta(t):
    return k1*t*t + k2*t + fi0

def chirp(t):
    return np.exp(1j*2*np.pi*theta(t))




#**************segunda parte*****************#
"""
c=chirp(x)

plt.plot(x2,np.pad(c,(1000,1000),'constant'))
plt.title("Chirpcorrida")
plt.ylabel("chirp")
plt.xlabel("tiempo")
plt.grid(True)
plt.show()


plt.plot(x,chirp(x))
plt.title("Chirp")
plt.ylabel("chirp")
plt.xlabel("tiempo")
plt.grid(True)
plt.show()
"""

#*******************OCHO************************

c=chirp(x)
c1=chirp(x)
cp=np.pad(c,(1000,1000),'constant')

correlacion=np.correlate(c1,cp, mode='valid')

plt.figure()
plt.plot(x4,correlacion)
plt.title("correlacion python")
plt.grid(True)
plt.show()



#********************NUEVE**********************
c=chirp(x)
c1=chirp(x)
cp=np.pad(c,(1000,1000),'constant')

dftcorrida = fft(cp,3000)/(len(cp)/2.0)  # fft de 2048 puntos 8192

dftnegconj = np.conj(fft(c1,3000)/(len(c1)/2.0))  # fft de 2048 puntos 8192

proddft = dftcorrida * dftnegconj

ifftcorrelacion = np.abs(ifft(proddft))

plt.figure()
plt.plot(x3,ifftcorrelacion)
#plt.plot(x2,np.pad(c,(1000,1000),'constant'),dashes=[1, 9])

plt.title("conjugo despues")
plt.grid(True)
plt.show()

#********

plt.figure()
plt.plot(x3,ifftcorrelacion, label="correlacion")
#plt.plot(x2,np.pad(c,(1000,1000),'constant'),dashes=[1, 9])

plt.axis([19.5e-6, 20.5e-6, 0,0.0017])
plt.axvline(x=20e-6+1/(19e6), color='#2ca02c', alpha=0.5, label="1/BW")
plt.axvline(x=20e-6+1/(-19e6), color='#2ca02c', alpha=0.5)
plt.title("jhjh")
plt.legend()
plt.grid(True)
plt.show()


#****************************