import PreProcessing as pre
import numpy as np
import matplotlib.pyplot as plt
import Segmentation as seg
import Spectrogram as sp

def main():
    prefix="F:/2020AccelEve/database/sampled_by_acc/400HZ/"
    word="key"
    person="wg"
    # prefix+person+'_'+word+'.tsv'
    # ====================read file======================
    tList, xList, yList, zList=pre.readFileGenByAcc(prefix+person+'_'+word+'_400.csv')
    t,x,y,z=(tList.copy(), xList.copy(), yList.copy(), zList.copy())
    # ====================preprocess======================
    tList, freqList, xList, yList, zList = \
        pre.standardize(tList, xList, yList, zList, highpass=120 / 500)
    # pre.showMap(freqList, xList, yList, zList, 'high passed', ylim=(0, 15))
    tList,xList, yList, zList = pre.reverseFFT(xList, yList, zList)
    pre.showMap(tList, xList, yList, zList, '120hz filter')

    zSmooth,tSmooth=seg.smooth(zList,tList)

    cuttingpoints = seg.findCuttingPoints(tSmooth,zSmooth,(0.8,0.25))

    # print(cuttingpoints)
    # ====================spectrogram======================
    t, freqList, x, y, z=pre.standardize(t,x,y,z,highpass=80/500)
    t,x,y,z=pre.reverseFFT(x, y, z)
    length = len(cuttingpoints)
    cnt=0

    print(cuttingpoints)
    print("=======================")
    for i in range(0, length, 2):
        x_one = x[cuttingpoints[i]+100:cuttingpoints[i + 1]+400]
        y_one = y[cuttingpoints[i]+100:cuttingpoints[i + 1]+400]
        z_one = z[cuttingpoints[i]+100:cuttingpoints[i + 1]+400]
        specX, specY, specZ = sp.generateMap(x_one, y_one, z_one)
        sp.generateRGB(specX, specY, specZ, prefix+person+'_acc_'+word+ str(cnt)+'.png')
        cnt+=1


if __name__=="__main__":
    main()