import os
from math import sqrt

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile


def readAudio(fileName):
    sample_rate, audio = wavfile.read(fileName)
    return sample_rate, audio

def smooth(zList):
    zList_tmp=zList
    length = len(zList)
    config1 = 20000
    config2 = 1000
    config3=50
    newSize = config1 + config2 +config3- 3
    # tSmooth = tList[:]
    zSmooth = [0 for i in range(0, length - newSize)]
    zList_tmp[:] = np.abs(zList_tmp)
    zSmooth[:] = np.convolve(zList_tmp, np.ones(config1) / config1, mode='valid')
    zSmooth[:] = np.convolve(zSmooth, np.ones(config2) / config2, mode='valid')
    zSmooth[:] = np.convolve(zSmooth, np.ones(config3) / config3, mode='valid')
    return zSmooth

def segment(zSmooth, para=(0.5,0.25)):
    zSmooth = np.array(zSmooth)
    Mmax = np.max(zSmooth)
    Mmin = np.min(zSmooth)
    leng = zSmooth.size
    thres = para[0] * Mmin + para[1] * Mmax
    cuttingpoints = []
    # =========================
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    # plt.xticks(np.linspace(0, 35000, 10))
    ax.set_title('cuttingpoints', fontsize=20)
    ax.plot(zSmooth, color='blue')
    ax.axhline(y=thres,xmin=0,xmax=1e6,color='red')
    # ========================
    for i in range(0, leng - 1, 1):
        if zSmooth[i] <= thres and zSmooth[i + 1] > thres:
            cuttingpoints.append(i+2500)
        elif zSmooth[i] >= thres and zSmooth[i + 1] < thres:
            cuttingpoints.append(i+1000)

    ax.scatter(cuttingpoints,zSmooth[cuttingpoints], color='purple', linewidths=1)
    fig.show()
    return cuttingpoints


def generateSoundMapFromWavFile(sample_rate, audio):
    nperseg = 6144  # the length of the windowing segments
    noverlap = 5760
    Fs = sample_rate  # the sampling frequency
    freqZ, ts,specZ= signal.stft(audio,nperseg=nperseg, fs=sample_rate, noverlap=noverlap,window="hann",detrend=False)
    specZ=np.abs(specZ)

    # plt.pcolormesh(ts, freqZ, specZ)
    #
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    return specZ


def generateRGBOfAudio(specZ, pic_name):
    # TODO: try x y z combination
    outpath = "F:/2020AccelEve/database/sound/_wav/rgb"#TODO:输出文件夹
    ROW = specZ.shape[0]
    COL = specZ.shape[1]
    B_z = np.zeros(shape=specZ.shape)
    for i in range(ROW):
        for j in range(COL):
            B_z[i, j] = sqrt(specZ[i, j])

    maxi = np.max(B_z)
    mini = np.min(B_z)

    B_z = np.ceil((255 / (maxi - mini)) * (B_z - mini))
    print(np.mean(B_z))

    img = np.zeros(shape=(ROW, COL, 3))
    img[:, :, 2] = np.zeros(shape=(ROW, COL))
    img[:, :, 1] = np.zeros(shape=(ROW, COL))
    img[:, :, 0] = B_z
    tmpStr = outpath + "/" + pic_name[:-4] + ".png"
    cv2.imwrite(tmpStr, img)
    # cv2.imshow('rgb',img)
    # cv2.waitKey()


if __name__ == "__main__":
    # TODO:输入文件夹（wav文件）
    path = "F:/2020AccelEve/database/sound/_wav"
    for fileName in os.listdir(path):
            if not os.path.isdir(path+'/'+fileName):
                STR = path + "/" + fileName
                sample_rate, audio= readAudio(STR)
                tempAudio=np.abs(audio)
                # ===========================================
                tempAudio=smooth(tempAudio)
                cuttingPoints=segment(tempAudio,para=(0.8,0.05))
                # ===========================================
                for i in range(0,len(cuttingPoints)-1,2):
                    clip=audio[cuttingPoints[i]:cuttingPoints[i+1]]
                    spec = generateSoundMapFromWavFile(sample_rate, clip)
                    generateRGBOfAudio(spec, pic_name=(str)(i)+fileName)


