from statsmodels import robust

import PreProcessing as pre
import Spectrogram as sp
from matplotlib import pyplot as plt
import math
import pandas as pd
import numpy as np
from scipy import stats


# =======time domain========
def getMean(z):
    result=np.mean(z)
    return result

def getStdDev(z):
    result=np.std(z)
    return result

#average deviation
def getAveDev(z):
    result=np.sum(np.abs(z-np.mean(z)))/len(z)
    return result

def getSkewness(z):
    return stats.skew(z)


def getKurtosis(z):
    return stats.kurtosis(z)

def getRMS(z):
    return np.sqrt(np.mean(z**2))

# =======Frequency domain========
def getSpecStdDev(fz,freqList):
    zm=fz
    zf=freqList
    d=np.sqrt(    np.sum( (zf**2)*zm )/np.sum(zm)       )
    return d

def getSpecCentroid(fz,freqList):
    zm = fz
    zf = freqList
    d=getSpecStdDev(fz,freqList)
    C= np.sum(   zf*zm   )/np.sum(zm)
    return C

def getSpecSkewness(fz,freqList):
    zm = fz
    zf = freqList
    C=getSpecCentroid(fz,freqList)
    d=getSpecStdDev(fz,freqList)
    gamma=(np.sum(  ((zm-C)**3) *zm ))/d**3
    return gamma

def getSpecKurt(fz,freqList):
    zm = fz
    zf = freqList
    C = getSpecCentroid(fz, freqList)
    d = getSpecStdDev(fz, freqList)
    beta=(np.sum(  ((zm-C)**4 ) *zm))/d**4 -3
    return beta

def getSpecCrest(fz,freqList):
    zm = fz
    zf = freqList
    C = getSpecCentroid(fz, freqList)

    CR=np.max(zm)/C
    return CR



def preprocess():
    word="code"
    person="wg_"

    # TODO: set picture to the same size regardless of its length
    # tList, xList, yList, zList = pre.readFileGenByAcc("./hotwords_rsc/"+person+word+".csv")
    tList, xList, yList, zList=pre.readFile("./S9one100five.tsv")
    tList, freqList, xList, yList, zList = pre.standardize(tList, xList, yList, zList, highpass=5 / 1000)
    length=len(tList)
    # fig,ax=plt.subplots(3,1)
    # ax[0].plot(freqList[0:round(length / 2)], xList[0:round(length / 2)], color='red')
    # ax[1].plot(freqList[0:round(length / 2)], yList[0:round(length / 2)], color='green')
    plt.plot(freqList[0:round(length/2)],np.abs(zList[0:round(length/2)]),color='blue')

    plt.show()
    # pre.showFreqMap(freqList[0:round(length/2)],zList[0:round(length/2)])
    # pre.showFreqMap(freqList[0:round(length/2)], xList[0:round(length/2)],title="X-Freq map",color='red')
    # pre.showFreqMap(freqList[0:round(length/2)], yList[0:round(length/2)],title="Y-Freq map",color='green')

def extract():
    word = "code"
    person = "ty_"

    prefix="F:/2020AccelEve/database/fixed_rate/S9/"
    filename="S9one"
    testFreq=100
    testCase=1
    postfix=".tsv"
    for i in range(4):
        testCase=1
        for j in range(5):
            STR=prefix+filename+(str)(testFreq)+'_'+(str)(testCase)+postfix
            # TODO: set picture to the same size regardless of its length
            tList, xList, yList, zList=pre.readFile(STR)
            # tList, xList, yList, zList = pre.readFileGenByAcc("./hotwords_rsc/" + person + word + ".csv")
            # =============time domain==================
            t=[]
            t.append(getMean(zList))
            t.append(getStdDev(zList))
            t.append(getKurtosis(zList))
            t.append(getSkewness(zList))
            t.append(getAveDev(zList))
            t.append(getRMS(zList))
            t=np.array(t)
            # =================freq domain==================
            f=[]
            tList, freqList, xList, yList, zList = pre.standardize(tList, xList, yList, zList, highpass=10 / 1000)
            length=len(zList)
            zList=zList[0:round(length/2)]
            freqList=freqList[0:round(length/2)]
            f.append(getSpecStdDev(np.abs(zList),freqList))
            f.append(getSpecCentroid(np.abs(zList),freqList))
            f.append(getSpecSkewness(np.abs(zList),freqList))
            f.append(getSpecKurt(np.abs(zList),freqList))
            f.append(getSpecCrest(np.abs(zList),freqList))
            f.append("Nothing")
            f=np.array(f)

            dic={'FreqDomain':f,'TimeDomain':t}
            DF=pd.DataFrame(data=dic)

            savedName=prefix+"FeatureVector/"+filename+(str)(testFreq)+'_'+(str)(testCase)+'.csv'
            DF.to_csv(path_or_buf=savedName)
            # DF.to_excel(excel_writer='./test.xlsx')
            # print(DF)
            testCase += 1
        testFreq += 100


if __name__=="__main__":
    # preprocess()
    extract()