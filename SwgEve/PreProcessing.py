import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import signal

def readFileGenByAcc(FileName):
    csv_reader=pd.read_csv(FileName)

    xList = csv_reader.iloc[:,0].values
    yList = csv_reader.iloc[:,1].values
    zList = csv_reader.iloc[:,2].values
    length=len(xList)
    tList=np.linspace(0,length/500*1000,length,dtype=int)
    return tList, xList, yList, zList

def readFile(FileName):
    tsv_reader = pd.read_csv(FileName, sep='\t', index_col=0, header=None)
    tList = tsv_reader[1].values
    # for i in range(len(tList)):
    #     tList[i]=(int)((float)(tList[i][7:])*1000)
    xList = tsv_reader[2].values
    yList = tsv_reader[3].values
    zList = tsv_reader[4].values
    return tList, xList, yList, zList


def showMap(tList, xList, yList, zList, title='Original map', ylim=None):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), sharey=True)
    if ylim is not None:
        plt.ylim(ylim)
    ax[0].set_title(title, fontsize=20)
    colors = ['red', 'green', 'blue']
    lists = [xList, yList, zList]
    for i in range(3):
        ax[i].plot(tList, lists[i], color=colors[i])
    fig.show()
def showFreqMap(freqList, zList, title='Z-Freq map',color='blue'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 12), sharey=True)
    # plt.xticks(np.linspace(0,25000,10))
    ax.set_title(title, fontsize=20)
    # ax.plot(tList, y, color='red')
    # plt.xlim((-500,500))
    # plt.ylim((8,11))
    plt.xticks(np.arange(-500,500,20))
    # ax.yticks(np.arange(7,12,0.5))
    ax.plot(freqList, np.abs(zList), color=color)
    fig.show()


def interp(tList, xList, yList, zList):  # 'slinear' interpolation
    newTimeSet = np.linspace(0, tList[-1], tList[-1])
    fX = interpolate.interp1d(tList, xList, kind='slinear')
    resultX = fX(newTimeSet)
    fY = interpolate.interp1d(tList, yList, kind='slinear')
    resultY = fY(newTimeSet)
    fZ = interpolate.interp1d(tList, zList, kind='slinear')
    resultZ = fZ(newTimeSet)
    return newTimeSet, resultX, resultY, resultZ


def highPassFilter(xList, yList, zList, thresRate):
    # number of samples
    length = xList.size
    # thres=(int)(length*thresRate)
    '''   
    normalize, 
    this does not need 'absolute value' as ifft needs it keep the way it used to be
    '''
    fs=1000
    freqsX,tx,resultX, = signal.stft(xList,fs,nperseg=128,noverlap=120,window="hann",boundary=None,padded=False,return_onesided=True)
    freqsY,ty,resultY = signal.stft(yList,fs,nperseg=128,noverlap=120,window="hann",boundary=None,padded=False,return_onesided=True)
    freqsZ,tz,resultZ = signal.stft(zList,fs,nperseg=128,noverlap=120,window="hann",boundary=None,padded=False,return_onesided=True)
    thres = np.ceil(resultZ.shape[0] * thresRate)
    # resultX = np.fft.fft(xList)
    # resultY = np.fft.fft(yList)
    # resultZ = np.fft.fft(zList)
    '''
    freqs above 1000/2, are symmetric to those below. 
    However, we dont do the cutoff here, 
    otherwise the timeline(tList) would be confusing
    '''

    for i in range(0,resultZ.shape[0]):
        if i <= thres:
            resultX[i,:] = 0
            resultY[i,:] = 0
            resultZ[i,:] = 0
    #===========debug=========
    # fig,ax=plt.subplots()
    # plt.title("resultZ")
    # ax.plot(freqList,np.real(zList))
    # fig.show()

    return freqsZ, resultX, resultY, resultZ

def reverseFFT(xList, yList, zList):
    fs=1000
    t,resultX = signal.istft(xList,fs,nfft=128,noverlap=120)
    t,resultY = signal.istft(yList,fs,nfft=128,noverlap=120)
    t,resultZ = signal.istft(zList,fs,nfft=128,noverlap=120)
    # resultX = np.abs(np.fft.ifft(xList))
    # resultY = np.abs(np.fft.ifft(yList))
    # resultZ = np.abs(np.fft.ifft(zList))
    # TODO: test if the cause is ABS
    return t,resultX,resultY,resultZ

def standardize(tList, xList, yList, zList, highpass=-1):
    '''
    used to standardize & interpolate and maybe high-pass filter data
    :param tList: timestamp gathered from data
    :param xList:
    :param yList:
    :param zList:
    :param highpass: threshold rate
    :return: fList is the list of Frequencies
    '''
    lists = [xList, yList, zList]
    tList-=tList[0]
    # zero mean
    for i in range(0, 3):
        lists[i][:] = np.subtract(lists[i],np.mean(lists[i]))/np.var(lists[i])

    tList, xList, yList, zList = interp(tList, xList, yList, zList)
    # remove initialing time(invalid sound),which in data714 is the first 4 seconds
    tList = tList[:-50]
    xList = xList[:-50]
    yList = yList[:-50]
    zList = zList[:-50]
    if highpass >= 0:
        fList, xList, yList, zList = highPassFilter(xList, yList, zList,highpass)
        return tList, fList, xList, yList, zList
    return tList, xList, yList, zList


if __name__ == '__main__':
    # name="test100one"
    prefix = "F:/2020AccelEve/database/sampled_by_s8/"
    word = "secret"
    person = "wg"
    tList, xList, yList, zList = readFile(prefix+person+'_s8'+'_'+word+'.tsv')
    showMap(tList, xList, yList, zList, title="s9_swg_origin")
    tList, freqList,xList, yList, zList = standardize(tList, xList, yList, zList, highpass=10/1000)
    xList, yList, zList = reverseFFT(xList, yList, zList)
    showMap(tList,xList,yList, zList,title="s9")

