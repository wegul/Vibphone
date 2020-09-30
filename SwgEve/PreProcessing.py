import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate


def readFile(FileName):
    tsv_reader = pd.read_csv(FileName, sep='\t', index_col=0, header=None)
    tList = tsv_reader[1].values
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
def showZMap(tList, zList, title='Original map'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), sharey=True)
    # plt.xticks(np.linspace(0,25000,10))
    ax.set_title(title, fontsize=20)
    # ax.plot(tList, y, color='red')
    ax.plot(tList, zList, color='blue')
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
    len = xList.size
    thres=(int)(len*thresRate)
    '''   
    normalize, 
    this does not need 'absolute value' as ifft needs it keep the way it used to be
    '''
    resultX = np.fft.fft(xList)
    resultY = np.fft.fft(yList)
    resultZ = np.fft.fft(zList)
    '''
    freqs above 1000/2, are symmetric to those below. 
    However, we dont do the cutoff here, 
    otherwise the timeline(tList) would be confusing
    '''
    # resultX = resultX[range(0,(int)(len / 2))]
    # resultY = resultY[range(0,(int)(len / 2))]
    # resultZ = resultZ[range(0,(int)(len / 2))]

    # due to Nyquist, we can only pick freqs below 1000/2
    freqs = np.fft.fftfreq(len,0.001)
    for i in range(0,len):
        if i <= thres or len - thres <= i:
        # if i<=thres:
            resultX[i] = 0
            resultY[i] = 0
            resultZ[i] = 0
    #===========debug=========
    # fig,ax=plt.subplots()
    # plt.title("resultZ")
    # ax.plot(freqList,np.real(zList))
    # fig.show()

    return freqs, resultX, resultY, resultZ

def reverseFFT(xList, yList, zList):
    resultX = np.real(np.fft.ifft(xList))
    resultY = np.real(np.fft.ifft(yList))
    resultZ = np.real(np.fft.ifft(zList))
    return resultX,resultY,resultZ

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
    tList = tList[900:-2000]
    xList = xList[900:-2000]
    yList = yList[900:-2000]
    zList = zList[900:-2000]
    if highpass >= 0:
        fList, xList, yList, zList = highPassFilter(xList, yList, zList,highpass)
        return tList, fList, xList, yList, zList
    return tList, xList, yList, zList


if __name__ == '__main__':
    tList, xList, yList, zList = readFile('src/handhold928ty1six.tsv')
    # showMap(tList, xList, yList, zList, 'orig')
    # 713time==21593ms
    # 714time==25039ms
    tList,freqList, xList, yList, zList = standardize(tList, xList, yList, zList, highpass=85/1000)
    xList, yList, zList=reverseFFT(xList, yList, zList)
    # showMap(tList, xList, yList, zList,title='interpolated',ylim=(-0.5,0.5))
    # tList,xList, yList, zList=highPassFilter(xList, yList, zList)
    showMap(tList, xList, yList, zList, '100hz high passed swg',ylim=(-0.02,0.02))
