"""
Compresión en rango La técnica de compresión de pulso se aplica a cada fila de la matriz,
 de modo tal de obtener una imagen comprimida en rango, es decir,
 una imagen SAR con alta resolución en la dirección de rango.

● Ejercicio 10. Cargar en memoria la matriz contenida en el archivo SAR_data_sint.mat.
Reutilizando lo desarrollado en el ejercicio (9), implemente un ciclo para focalizar la imagen
en la dirección de rango, es decir, realizar la correlación de cada fila de la matriz con el chirp emitido.
Visualice el módulo de la imagen antes y luego de la compresión, donde debería ver claramente la compresión
en rango (ver figura 7 para una guía ilustrativa). Una vez que verifique el correcto funcionamiento,
aplique el mismo algoritmo a la imagen SARAT.

La técnica que se utiliza para enfocar la imagen en la dirección de acimut se describe
en la sección siguiente, y es a la que se debe la denominación de “apertura sintética” a este tipo de radares


"""

import numpy as np
import math as m
import scipy as sci
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftshift
import pylab as pl
from numpy import array
from numpy import empty
import scipy.io as sio
data1 = sio.loadmat('SAR_data_sint.mat')
#print(data)
data=data1['data_sint']


#data1 = sio.loadmat('SAR_data_sarat.mat')
#print(data1)
#data=data1['data']


k1 = 0.19 * 10e12
k2 = -1.9 * 10e6
fi0 = 0

fs = 50e6

x=np.r_[0:(10e-6-1/fs):1/(fs)]
print(len(x))



def theta(t):
    return k1*t*t + k2*t + fi0

def chirp(t):
    return np.exp(1j*2*np.pi*theta(t))

c=chirp(x)

dftchirp_rgconj= np.conj(fft(c,2048)/(len(c)/2.0))

"""
N=len(data1['data_sint'])
print(N)
data=data1['data_sint']
print(data)
N=len(data)
print(data.shape)
"""
print(data.shape)
print(data.shape[0])
print(data.shape[1])
data_rc1=np.zeros([4250,0])
print("dararc")
print(data_rc1.shape)
print(data_rc1)
nfil=data.shape[0]
ncol=data.shape[1]
data_rc=data
print(data_rc.shape)
for i in range(nfil):
    d = data[i]
    dftdata = fft(d, 2048) / (len(d) / 2.0)
    dftdata_rc=dftchirp_rgconj*dftdata
    #data_rc.append(np.abs(ifft(dftdata)))######
    data_rc[i]=np.abs(ifft(dftdata))
    print("DATAI",data_rc[i])
print(data[2125][1024])
print(data_rc[2125][1024])
print(data_rc.shape)

"""
data_comp_tras = []

for j in range(ncol):
    columna_actual = []
    for i in range(nfil):
        columna_actual.append(data_rc[i][j])
    #columna_actual=fft(columna_actual,4250)
    data_comp_tras.append(columna_actual)#la tengo que traspner

#la traspongo
data_comp = []
for i in range(nfil):
    columna_actual = []
    for j in range(ncol):
        columna_actual.append(data_comp_tras[j][i])
    data_comp.append(columna_actual)

data_comp = np.asmatrix(data_comp)
print(data_comp.shape)
print(data_comp)


window2 = signal.hann(50)
f, t, Sxx = signal.spectrogram(data_rc, fs, window2, noverlap=35, return_onesided=False)  #
f = np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx, axes=0)
plt.figure()
plt.pcolormesh(t, f, (Sxx))  # expectograma en escala logaritmica
plt.show()

"""