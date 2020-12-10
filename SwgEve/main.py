import PreProcessing as pre
import numpy as np
import matplotlib.pyplot as plt
import Segmentation as seg
import Spectrogram as sp

if __name__=='__main__':
    word="pswd"
    person="ty"

    # ====================read file======================
    tList, xList, yList, zList=pre.readFileGenByAcc("./rsc/"+person+'_'+word+".csv")
    # tList, xList, yList, zList=pre.readFile('src/handhold1013one.tsv')
    t,x,y,z=(tList, xList, yList, zList)
    # ====================preprocess======================
    tList, freqList, xList, yList, zList = \
        pre.standardize(tList, xList, yList, zList, highpass=120 / 1000)
    # pre.showMap(freqList, xList, yList, zList, 'high passed', ylim=(0, 15))
    xList, yList, zList = pre.reverseFFT(xList, yList, zList)
    # pre.showMap(tList, xList, yList, zList, '120hz filter')

    # specX, specY, specZ = sp.generateMap(xList, yList, zList)
    zSmooth,tSmooth=seg.smooth(zList,tList)

    # pre.showZMap(tSmooth, zSmooth, 'smoothed_conv')
    cuttingpoints = seg.findCuttingPoints(tSmooth,zSmooth,(0.8,0.2))

    # print(cuttingpoints)
    # ====================spectrogram======================
    t, freqList, x, y, z=pre.standardize(t,x,y,z,highpass=85/1000)
    x,y,z=pre.reverseFFT(x, y, z)
    length = len(cuttingpoints)
    cnt=0
    for i in range(0, length, 2):
        x_one = x[cuttingpoints[i]:cuttingpoints[i + 1]+100]
        y_one = y[cuttingpoints[i]:cuttingpoints[i + 1]+100]
        z_one = z[cuttingpoints[i]:cuttingpoints[i + 1]+100]
        specX, specY, specZ = sp.generateMap(x_one, y_one, z_one)
        sp.generateRGB(specX, specY, specZ, './'+person+'/'+person+'_' + str(cnt) +word+'.png')
        cnt+=1
