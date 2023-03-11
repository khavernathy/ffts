import numpy as np
import scipy as sp
import pyfftw
from time import time
from multiprocessing import Pool, cpu_count

# numpy fft
def npfft(x):
    return np.fft.fft(x,axis=0)


# scipy fft
def spfft(x):
    return sp.fft.fft(x,axis=0)


# pyfftw fft
def pffft(x):
    a = pyfftw.builders.fft(x, threads=cpu_count())
    return a()


# discrete FT by O(N^2) loop
def dft(x):
    xhat = np.zeros(x.shape) + np.zeros(x.shape)*1j
    N = len(x)
    for k in range(N):
        for n in range(N):
            xhat[k] += x[n] * np.exp(-1j * 2.0*np.pi * k * n / N)
    return xhat


# parallelized O(N^2) DFT
def get_dft_entry(x, k, N):
        out = 0.
        for n in range(N):
            out += x[n] * np.exp(-1j * 2.0*np.pi * k * n / N)
        return out
def dft_parallel(x):
    xhat = np.zeros(x.shape) + np.zeros(x.shape)*1j
    pool = Pool(cpu_count())
    N = len(x)
    results = []
    for k in range(N):
        r = pool.apply_async(get_dft_entry, args=(x, k, N))
        results.append(r)
    pool.close()
    pool.join()
    for i,r in enumerate(results):
        xhat[i] = r.get()
    return xhat


# DFT by matrix multiplication
def dft_matrix(x):
    N = len(x)
    k = np.arange(N).reshape(1, N)
    n = np.arange(N).reshape(N, 1)
    return np.sum(x[...] * np.exp(-1j *2.0*np.pi * k*n / N), axis=1)


# DFT by interleaving even and odd terms
def dft_interleave(x):
    N = len(x)
    if N%2 == 0:
        e = x[::2]
        o = x[1::2]
        E = dft_matrix(e)
        O = dft_matrix(o)

        X = np.zeros((N)) + np.zeros((N))*1j
        for k in np.arange(int(N/2)):
            twid = np.exp(-2.0*np.pi * 1j * k / N)
            X[k] = E[k] + twid * O[k]
            X[k+int(N/2)] = E[k] - twid * O[k]
        return X
    else:
        return None
        raise Exception("input array is not divisible by 2")


# radix-2 DIT FFT, a la Cooley-Tukey
def is_power_of_2(N):
    return (N & (N - 1) == 0) and N != 0
def dft_radix2(x):
    N = len(x)
    if N==1:
        return x
    elif is_power_of_2(N):
        E = dft_radix2(x[::2])
        O = dft_radix2(x[1::2])

        X = np.zeros((N)) + np.zeros((N))*1j
        for k in np.arange(int(N/2)):
            twid = np.exp(-2.0*np.pi * 1j * k / N)
            X[k] = E[k] + twid * O[k]
            X[k+int(N/2)] = E[k] - twid * O[k]
        return X
    else:
        return None
        raise Exception("input array is not of power-of-2 length.")


# radix-2, but faster by avoiding k loop.
def dft_radix2_faster(x):
    N = len(x)
    if N==1:
        return x
    elif is_power_of_2(N):
        E = dft_radix2(x[::2])
        O = dft_radix2(x[1::2])

        X = np.zeros((N)) + np.zeros((N))*1j
        halfn = int(N/2)
        krange = np.arange(halfn)
        twid = np.exp(-2.0*np.pi * 1j * krange / N)
        X[krange] = E[krange] + twid[krange] * O[krange]
        X[krange + halfn] = E[krange] - twid[krange] * O[krange]
        return X
    else:
        return None
        raise Exception("input array is not of power-of-2 length.")


#f = open("dft_timing4096-1000000.txt", "w")
#f.write("n numpy scipy pyfftw dft_loop dft_para_loop dft_matr dft_inter dft_radix2 dft_radix2f\n")

l_np_time=[]
l_sp_time=[]
l_pf_time=[]
l_dft_time=[]
l_para_time=[]
l_matr_time=[]
l_interleave_time=[]
l_rad2_time=[]
l_rad2f_time=[]

for sz in np.arange(1,25000):
    #x = np.arange(sz)
    x = (np.random.random((sz, 1)) + np.random.random((sz, 1))*1j).flatten()
    print(sz)

    start = time()
    xhnp = npfft(x)
    nptime = time()-start
    #print("numpy time: ", nptime)
    l_np_time.append((sz,nptime))

    start = time()
    xhsp = spfft(x)
    sptime = time()-start
    #print("scipy time: ", sptime)
    l_sp_time.append((sz,sptime))

    start = time()
    xhpf = pffft(x)
    pftime = time()-start
    #print("pyfftw time: ", pftime)
    l_pf_time.append((sz,pftime))
    """
    start = time()
    xhat = dft(x)
    dfttime=time()-start
    #print("dft manual loop time: ", dfttime)
    l_dft_time.append((sz,dfttime))

    start = time()
    xhat_p = dft_parallel(x)
    paratime = time()-start
    #print("dft parallel loop time: ", paratime)
    l_para_time.append((sz,paratime))

    start = time()
    xhat_m = dft_matrix(x)
    matrtime = time()-start
    #print("dft matrix time: ", matrtime)
    l_matr_time.append((sz,matrtime))

    start = time()
    xhat_int = dft_interleave(x)
    inttime = time()-start if xhat_int is not None else 0
    #print("dft interleave time:", inttime)
    l_interleave_time.append((sz,inttime))

    start = time()
    xhat_rad2 = dft_radix2(x)
    rad2time = time()-start if xhat_rad2 is not None else 0
    #print("dft radix2 time: ", rad2time)
    l_rad2_time.append((sz,rad2time))

    start=time()
    xhat_rad2f = dft_radix2_faster(x)
    rad2ftime = time()-start if xhat_rad2f is not None else 0
    #print("dft radix2 faster time: ", rad2ftime)
    l_rad2f_time.append((sz,rad2ftime))
    """

    # compare equality across different methods
    """
    print("np vs. sp: ", np.allclose(xhnp, xhsp))
    print("np vs. pf: ", np.allclose(xhnp, xhpf))
    print("np vs. dft: ", np.allclose(xhnp, xhat))
    print("np vs. dft_p: ", np.allclose(xhnp, xhat_p))
    print("np vs. dft_m: ", np.allclose(xhnp, xhat_m))
    print("np vs. dft_int: ", np.allclose(xhnp, xhat_int))
    print("np vs. dft_r2: ", np.allclose(xhnp, xhat_rad2))
    print("np vs. dft_r2_fast: ", np.allclose(xhnp, xhat_rad2f))
    """

names = [i for i in dir() if i.startswith('l') and 'time' in i]
np.savez('1e6.npz', timings=np.array([eval(i) for i in names]), names=names)

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

data = np.load('1e6.npz', allow_pickle=True)
plt.figure()
for i in range(len(data['names'])):
    inds = np.where(data['timings'][i,:,1] != 0, True, False)
    plt.plot(data['timings'][i, inds, 0], data['timings'][i, inds, 1], label=data['names'][i])
plt.legend()
plt.show(block=True)

a=3
