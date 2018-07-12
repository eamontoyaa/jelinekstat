# import the functions
from jelinekstat.jelinekstat import tensorStat
import matplotlib.pyplot as plt

# Input data
plt.ion()

plt.plot([1,2,3,4,5,5,4,1,3,5,5,6,6])
plt.show()
sample = [[1.02327, 1.02946, 0.94727, -0.01495, -0.03599, -0.05574],
          [1.02315, 1.01803, 0.95882, -0.00924, -0.02058, -0.03151],
          [1.02801, 1.03572, 0.93627, -0.03029, -0.03491, -0.06088],
          [1.02775, 1.00633, 0.96591, -0.01635, -0.04148, -0.02006],
          [1.02143, 1.01775, 0.96082, -0.02798, -0.04727, -0.02384],
          [1.01823, 1.01203, 0.96975, -0.01126, -0.02833, -0.03649],
          [1.01486, 1.02067, 0.96446, -0.01046, -0.01913, -0.03864],
          [1.04596, 1.01133, 0.94271, -0.01660, -0.04711, -0.03636]]
confLevel = 0.95

jelinekStatsSummary, stereonetPlot = tensorStat(
        sample, confLevel=0.95, want2plot=True, plotName='shortSCRoutcome',
        ext='pdf')
stereonetPlot.show()

