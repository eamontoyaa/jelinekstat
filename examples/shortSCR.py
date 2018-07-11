# import the functions
from jelinekstat.tools import dataFromFile
from jelinekstat.jelinekstat import tensorStat

sample, numTensors = dataFromFile('exampledata.txt')

jelinekStatsSummary, stereonetPlot = tensorStat(
        sample, confLevel=0.95, want2plot=True, plotName='shortSCRoutcome',
        ext='pdf')
stereonetPlot.show()
