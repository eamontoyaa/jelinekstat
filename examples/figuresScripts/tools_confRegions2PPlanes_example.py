# import the functions
from numpy import cov
from jelinekstat.tools import dataFromFile, confRegions2PPlanes
from jelinekstat.jelinekstat import normalizeTensors,\
    meantensor, covMtx2PPlane, localCovMtxs, eigVectsRegions

sample, numTensors = dataFromFile('../examples/exampledata.txt')
normTensors = normalizeTensors(sample)
meanTensorVect, meanTensorMtx, numTensors = meantensor(
    normTensors, normalized=True)
covMtx = cov(normTensors.T, bias=False)
pCovMtx = covMtx2PPlane(
    covMtx, meanTensorVect, numTensors)
W, eigVal_W, eigVec_W = localCovMtxs(
    meanTensorVect, pCovMtx)
majorAxis, minorAxis, theta = eigVectsRegions(
    W, eigVal_W, eigVec_W, numTensors, confLvl=0.95,
    estimate=True)
x, y, fig = confRegions2PPlanes(majorAxis, minorAxis, theta,
                                want2plot=True, confLvl=0.95)
fig.show()
