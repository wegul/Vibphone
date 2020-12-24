import PreProcessing as pre
import Segmentation as seg
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
import cv2
def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def generateSoundMapFromWavFile(fileName):

    # audio = audiosegment.from_file(fileName)

    sample_rate, audio = wavfile.read(fileName)
    frequencies, times, spectrogram = log_specgram(audio,sample_rate)
    spectrogram=np.transpose(spectrogram)
    plt.pcolormesh( times,frequencies, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return spectrogram

def generateMap(xList, yList, zList,name="Z AXIS"):
    # global tList, xList, yList, zList

    NFFT = 128  # the length of the windowing segments
    noverlap=120
    Fs = 1000  # the sampling frequency
    fig,(plot_X,plot_Y,plot_Z)=plt.subplots(ncols=3)

    fX, tX, specX = signal.stft(xList,fs=Fs,window='hann',nperseg=NFFT,noverlap=noverlap,detrend=False)
    fY, tY, specY = signal.stft(yList,fs=Fs,window='hann',nperseg=NFFT,noverlap=noverlap,detrend=False)
    fZ, tZ, specZ = signal.stft(zList,fs=Fs,window='hann',nperseg=NFFT,noverlap=noverlap,detrend=False)

    specX = np.abs(specX)
    specY = np.abs(specY)
    specZ = np.abs(specZ)

    plot_X.pcolormesh(tX, fX, specX)
    plot_Y.pcolormesh(tY, fY, specY)
    plot_Z.pcolormesh(tZ, fZ, specZ)

    # specX,ts,freqsX,picX=plot_X.specgram(xList,NFFT=NFFT,Fs=Fs,noverlap=noverlap)
    # specY,ts,freqsY,picY=plot_Y.specgram(yList, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
    # specZ,ts,freqZ,picZ=plot_Z.specgram(zList, NFFT=NFFT, Fs=Fs, noverlap=noverlap)

    plot_X.set_title("X AXIS")
    plot_X.set_xlabel("time")
    plot_X.set_ylabel("frequency")
    plot_Y.set_title("Y AXIS")
    plot_Y.set_xlabel("time")
    plot_Y.set_ylabel("frequency")
    #TODO: change it back!!!! return!!!
    plot_Z.set_title(name)
    plot_Z.set_xlabel("time")
    plot_Z.set_ylabel("frequency")
    plt.show()
    return specX, specY, specZ


def generateRGB(specX, specY, specZ,name):
    R_x=np.zeros(shape=specX.shape)
    G_y=np.zeros(shape=specX.shape)
    B_z=np.zeros(shape=specX.shape)

    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            R_x[i,j]=math.sqrt(specX[i,j])
    maxi = np.max(R_x)
    mini = np.min(R_x)
    R_x = np.ceil((255 / (maxi - mini)) * (R_x - mini))

    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            G_y[i,j]=math.sqrt(specY[i,j])
    maxi = np.max(G_y)
    mini = np.min(G_y)
    G_y = np.ceil((255 / (maxi - mini)) * (G_y - mini))
    # print(np.mean(G_y))
    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            B_z[i,j]=math.sqrt(specZ[i,j])
            # B_z[i, j] = math.fabs(specZ[i, j])

    maxi=np.max(B_z)
    mini=np.min(B_z)

    B_z=np.ceil((255/(maxi-mini))*(B_z-mini))
    print(np.mean(B_z))

    img=np.zeros(shape=(specX.shape[0],specX.shape[1],3))
    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            img[i,j,2]=0
            img[i, j, 1] = 0
            img[i,j,0]=B_z[i,j]
    cv2.imwrite(name,img)
    # cv2.imshow('rgb',img)
    # cv2.waitKey()


def main():
    global tList, xList, yList, zList
    tList, xList, yList, zList = pre.readFile('siri_digits/siri_one_up_bass1.tsv')

    tList, freqList, xList, yList, zList = \
        pre.standardize(tList, xList, yList, zList, highpass=85/ 1000)

    xList, yList, zList = pre.reverseFFT(xList, yList, zList)
    # pre.showMap(tList, xList, yList, zList, '85hz filter')
    # zSmooth, tSmooth = seg.smooth(zList, tList)

    cuttingpoints=[1196, 2021, 3909, 4711, 6614, 7425, 9275, 10085, 11947, 12760, 14604, 15440]


    cnt=0
    length=len(cuttingpoints)
    for i in range(0,length,2):
        x_one=xList[cuttingpoints[i]:cuttingpoints[i+1]]
        y_one=yList[cuttingpoints[i]:cuttingpoints[i+1]]
        z_one=zList[cuttingpoints[i]:cuttingpoints[i+1]]
        specX,specY,specZ=generateMap(x_one,y_one,z_one)
        generateRGB(specX, specY, specZ,'siri_digits/siribassRGB'+str(cnt)+'two.png')
        cnt+=1



if __name__=='__main__':
    tList=[]
    xList=[]
    yList=[]
    zList=[]
    main()
