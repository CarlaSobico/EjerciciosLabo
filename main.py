import numpy as np
import math as m
import scipy as sci
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.fftpack import fftshift
import pylab as pl

k1 = 0.19 * 10e12
k2 = -1.9 * 10e6
fi0 = 0

fs = 50e6

#x=np.arange(N) / float(fs)
x=np.r_[0:(10e-6+(1/fs)):1/(fs)]
print(len(x));
#x=np.linspace(0,10e-6,M)

def theta(t):
    return k1*t*t + k2*t + fi0

def chirp(t):
    return np.exp(1j*2*np.pi*theta(t))


"""

plt.plot(x,np.real(chirp(x)))
plt.title("Parte real de chirp")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.grid(True)
plt.show()


plt.plot(x,np.imag(chirp(x)))
plt.title("Parte imaginaria de chirp")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.grid(True)
plt.show()

plt.figure(1)
plt.plot(x,np.unwrap(np.angle(chirp(x))), label="Ángulo chirp")
plt.plot(x,2*np.pi*theta(x),dashes=[9, 3], label="Función de fase")
plt.title("Función de fase")
plt.legend()
plt.ylabel("Fase")
plt.xlabel("Tiempo")
plt.grid(True)
plt.show()

plt.figure()
A = fft(chirp(x),8192)/(len(chirp(x))/2.0)  # fft de 2048 puntos 8192
#cuando hago la fft la amplitud de la transformada se ve afectada por la
#cantidad de puntos que tome, por lo tanto normalizo por la longitud de
#la venatana
freq =  np.linspace(-fs/2, fs/2, len(A))#va de -pi a pi
response = 20 * np.log10(np.abs(fftshift(A)))

plt.plot(freq,response, label="DFT"), plt.grid()
plt.axvline(x=19e6, color='#2ca02c', alpha=0.5, label="±19 MHz")
plt.axvline(x=-19e6, color='#2ca02c', alpha=0.5)
plt.title("(DFT)Respuesta en frecuencia de chirp")#arreglar label
plt.ylabel("Magnitud [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.legend()
plt.grid(True)
plt.show()
"""

'''
Compute and plot the spectrogram 
'''


"""
fs = 50e6  #nyquist
x1=np.r_[0:(10e-6+(1/fs)):1/(fs)]
window = signal.tukey(50) # ventana de Tukey de 256 muestras

plt.figure()
A = fft(window,8192)/(len(window)/2.0)
freq =  np.linspace(-20e6/2, 20e6a/2, len(A))#va de -pi a pi
response = 20 * np.log10(np.abs(fftshift(A)))
plt.plot(freq,response, label="DFT")
plt.title("(DFT)Respuesta en frecuencia de la ventana de Tukey")#arreglar label
plt.ylabel("Magnitud [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.show()
"""
"""
f, t, Sxx = signal.spectrogram(chirp(x1), fs, window,noverlap=35,return_onesided=False)#
f=np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx,axes=0)

plt.figure(1)
plt.suptitle("Espectograma con ventana de Tukey")
plt.subplot(2,1,1)
plt.pcolormesh(t/1e-6,f,Sxx) #espectograma en escala lineal
plt.title("En escala lineal")#arreglar label
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [microsec]')
plt.colorbar()
#plt.show()

plt.subplot(2,1,2)
plt.pcolormesh(t/1e-6,f,np.log10(Sxx))#expectograma en escala logaritmica
plt.title("En escala logaritmica")#arreglar label
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [microsec]')
plt.colorbar()
plt.show()
"""
#"""
window2 = signal.hann(50)
plt.figure()
A = fft(window2,8192)/(len(window2)/2.0)
freq =  np.linspace(-20e6/2, 20e6/2, len(A))#va de -pi a pi
response = 20 * np.log10(np.abs(fftshift(A)))
plt.plot(freq,response, label="DFT")
plt.title("(DFT)Respuesta en frecuencia de la ventana de Hanning")#arreglar label
plt.ylabel("Magnitud [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.show()
#"""
"""
plt.figure(2)
plt.suptitle("Espectograma con ventana de Hanning")
plt.subplot(2,1,1)
f, t, Sxx = signal.spectrogram(chirp(x1), fs, window2,noverlap=35,return_onesided=False)#
f=np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx,axes=0)
plt.pcolormesh(t/1e-6,f,Sxx) #espectograma en escala lineal
plt.title("En escala lineal")#arreglar label
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [microsec]')
plt.colorbar()
#plt.show()

plt.subplot(2,1,2)
plt.pcolormesh(t/1e-6,f,np.log10(Sxx))#expectograma en escala logaritmica
plt.title("En escala logaritmica")#arreglar label
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [microsec]')
plt.colorbar()
plt.show()
"""

"""
fs = 50e6  #nyquist
x1=np.r_[0:(10e-6+(1/fs)):1/(fs)]
window3 = signal.triang(50)

plt.figure()
A = fft(window3,8192)/(len(window3)/2.0)
freq =  np.linspace(-20e6/2, 20e6/2, len(A))#va de -pi a pi
response = 20 * np.log10(np.abs(fftshift(A)))
plt.plot(freq,response, label="DFT")
plt.title("(DFT)Respuesta en frecuencia de la ventana triangular")#arreglar label
plt.ylabel("Magnitud [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.show()
"""
"""
plt.figure(2)
plt.suptitle("Espectograma con ventana triangular")
plt.subplot(2,1,1)
f, t, Sxx = signal.spectrogram(chirp(x1), fs, window3,noverlap=35,return_onesided=False)#
f=np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx,axes=0)
plt.pcolormesh(t/1e-6,f,Sxx) #espectograma en escala lineal
plt.title("En escala lineal")#arreglar label
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [microsec]')
plt.colorbar()
#plt.show()

plt.subplot(2,1,2)
plt.pcolormesh(t/1e-6,f,np.log10(Sxx))#expectograma en escala logaritmica
plt.title("En escala logaritmica")#arreglar label
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [microsec]')
plt.colorbar()
plt.show()

"""
#*************************************

"""


fs = 50e6  #nonyquist
x2=np.r_[0:(10e-6+(1/fs)):1/(fs)]
window = signal.triang(50) # ventana de Tukey de 256 muestras
window2 = signal.hann(50)
plt.figure()
A = fft(window1,8192)/(len(window2)/2.0)
freq =  np.linspace(-20e6/2, 20e6/2, len(A))#va de -pi a pi
response = 20 * np.log10(np.abs(fftshift(A)))
plt.plot(freq,response, label="DFT")
plt.title("(DFT)Respuesta en frecuencia de la ventana de Hanning")#arreglar label
plt.ylabel("Magnitud [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True)
plt.show()
f, t, Sxx = signal.spectrogram(chirp(x2), fs, window,noverlap=0,return_onesided=False)#
f=np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx,axes=0)
plt.figure(1)
plt.subplot(2,2,1)
plt.pcolormesh(t,f,Sxx) #espectograma en escala lineal
plt.title("Espectograma en escala lineal con tukey")#arreglar label
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

plt.subplot(2,2,2)
#plt.show()
plt.pcolormesh(t,f,np.log10(Sxx))#expectograma en escala logaritmica
plt.title("Espectograma en escala logaritmica con tukey")#arreglar label
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.show()

plt.subplot(2,2,3)
f, t, Sxx = signal.spectrogram(chirp(x2), fs, window2,noverlap=0,return_onesided=False)#
f=np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx,axes=0)
plt.pcolormesh(t,f,Sxx) #espectograma en escala lineal
plt.title("Espectograma en escala lineal con hanning")#arreglar label
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.show()

plt.subplot(2,2,4)
plt.pcolormesh(t,f,np.log10(Sxx))#expectograma en escala logaritmica
plt.title("Espectograma en escala logaritmica con hanning")#arreglar label
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


#/************hacer una triangular******************/
"""