Use and Examples
================

Importing data from file
------------------------

In some cases the data to process is stored in external files. Due to those cases, this application software is able to import the data from an external tabulate ``.txt`` which contains the data of the randomly selected sample. It is done by using the function ``dataFromFile`` from the ``tools.py`` module.

Some conditions are required in the file format for importing it. It has to be structured as a :math:`(n \times 6)` array, where math:`n` is the number of tensors and each row contains a tensor in the vector form with the 6 components sorted like this order :math:`t_{11}, t_{22}, t_{33}, t_{12}, t_{23}, t_{13}`.

The ``exampledata.txt`` file has the following content: ::

    1.02327    1.02946    0.94727    -0.01495    -0.03599    -0.05574
    1.02315    1.01803    0.95882    -0.00924    -0.02058    -0.03151
    1.02801    1.03572    0.93627    -0.03029    -0.03491    -0.06088
    1.02775    1.00633    0.96591    -0.01635    -0.04148    -0.02006
    1.02143    1.01775    0.96082    -0.02798    -0.04727    -0.02384
    1.01823    1.01203    0.96975    -0.01126    -0.02833    -0.03649
    1.01486    1.02067    0.96446    -0.01046    -0.01913    -0.03864
    1.04596    1.01133    0.94271    -0.01660    -0.04711    -0.03636

And the minimum script for importing ``exampledata.txt`` is the following: ::

    # import the function
    from jelinekstat.tools import dataFromFile

    sample, numTensors = dataFromFile('exampledata.txt')


Examples
--------

Two ways for executing the application software via script are presented below.

Short script
^^^^^^^^^^^^

The first way is by using the function ``tensorStat`` from the
``jelinekstat.py`` module in a short script as follow ::
    
    from jelinekstat.jelinekstat import tensorStat

    # Input data
    sample = [[1.02327, 1.02946, 0.94727, -0.01495, -0.03599, -0.05574],
              [1.02315, 1.01803, 0.95882, -0.00924, -0.02058, -0.03151],
              [1.02801, 1.03572, 0.93627, -0.03029, -0.03491, -0.06088],
              [1.02775, 1.00633, 0.96591, -0.01635, -0.04148, -0.02006],
              [1.02143, 1.01775, 0.96082, -0.02798, -0.04727, -0.02384],
              [1.01823, 1.01203, 0.96975, -0.01126, -0.02833, -0.03649],
              [1.01486, 1.02067, 0.96446, -0.01046, -0.01913, -0.03864],
              [1.04596, 1.01133, 0.94271, -0.01660, -0.04711, -0.03636]]
    confLevel = 0.95

    # Performing the calculation all in one function.
    jelinekStatsSummary, stereonetPlot = tensorStat(
            sample, confLevel=0.95, want2plot=True, plotName='testForSCRshort',
            ext='pdf')
    stereonetPlot.show()

.. plot::

    from jelinekstat.tools import dataFromFile
    from jelinekstat.jelinekstat import tensorStat
    sample, numTensors = dataFromFile('exampledata.txt')
    jelinekStatsSummary, stereonetPlot = tensorStat(
            sample, confLevel=0.95, want2plot=True, plotName='test',
            ext='pdf')
    stereonetPlot.show()

Long script
^^^^^^^^^^^

The second way is by using all the code lines inside the same function used above in a much longer script as follow ::

    import numpy as np
    import matplotlib.pyplot as plt

    from jelinekstat.jelinekstat import normalizeTensors, meantensor, \
        covMtx2PPlane, eigValsIntervals, localCovMtxs, eigVectsRegions
    from jelinekstat.tools import getEigSorted, confRegions2PPlanes, \
        eigVects2PlgTrd, proyAllEllipses2LongLat, splitIterables

    # Input data.
    sample = [[1.02327, 1.02946, 0.94727, -0.01495, -0.03599, -0.05574],
              [1.02315, 1.01803, 0.95882, -0.00924, -0.02058, -0.03151],
              [1.02801, 1.03572, 0.93627, -0.03029, -0.03491, -0.06088],
              [1.02775, 1.00633, 0.96591, -0.01635, -0.04148, -0.02006],
              [1.02143, 1.01775, 0.96082, -0.02798, -0.04727, -0.02384],
              [1.01823, 1.01203, 0.96975, -0.01126, -0.02833, -0.03649],
              [1.01486, 1.02067, 0.96446, -0.01046, -0.01913, -0.03864],
              [1.04596, 1.01133, 0.94271, -0.01660, -0.04711, -0.03636]]
    confLevel = 0.95

    # PERFORMING THE CALCULATION STEP BY STEP.

    # Normalization of the imput sample
    sample = normalizeTensors(sample)

    # Mean tensor from normalized data and sample size (n)
    k, K, n = meantensor(sample, True)  # k (vector); K (matrix)

    # Eigenvalues (kK) and eigenvectors (pK) of the mean tensor
    kK, pK = getEigSorted(K)

    # Unbiased covariance matrix (V).
    V = np.cov(sample.T, bias=False)

    # Covariance matrix (V) in the system of the k's principal vectors (pV).
    pV = covMtx2PPlane(V, k, n)

    # Confidence intervals of eigenvalues of mean tensor (kIntervals).
    kIntervals = eigValsIntervals(pV, n, confLevel)

    # Local covariance matrices (W) in each P-plane of each confidence region.
    W, eigValW, eigVectW = localCovMtxs(k, pV)

    # Length and orientation of ellipses semi-axis.
    majorAxis, minorAxis, theta = eigVectsRegions(
            W, eigValW, eigVectW, n, confLevel)

    # Coordiantes of the three ellipses in each P-plane.
    x, y, PPlanePlots = confRegions2PPlanes(
            majorAxis, minorAxis, theta, True, confLevel)

    # Stereographic notation to plot the mean tensor's principal vectors (pK).
    pKPlg, pKTrd = eigVects2PlgTrd(k)  # Plg (plunge); Trd (trend)

    # (plunge,trend) notation to plot principal axis of all tensors.
    samplePlgTrd = list(map(eigVects2PlgTrd, sample))

    # (lon, lat) notation of each confidence region.
    kRegionsLong, kRegionsLat = proyAllEllipses2LongLat(x, y, k)

    # Summary of the Jelinek (1978) statistic proposal for 2nd-order tensors.
    jelinekStatSummary = {
            'k': k,
            'n': n,
            'k1': {'mean': kK[0], 'variability': kIntervals[0]},
            'k2': {'mean': kK[1], 'variability': kIntervals[1]},
            'k3': {'mean': kK[2], 'variability': kIntervals[2]},
            'p1': {'coords': pK[:, 0], 'plg': pKPlg[0], 'trd': pKTrd[0],
                   'majAx': majorAxis[0], 'minAx': minorAxis[0],
                   'incl': np.degrees(theta[0])},
            'p2': {'coords': pK[:, 1], 'plg': pKPlg[1], 'trd': pKTrd[1],
                   'majAx': majorAxis[1], 'minAx': minorAxis[1],
                   'incl': np.degrees(theta[1])},
            'p3': {'coords': pK[:, 2], 'plg': pKPlg[2], 'trd': pKTrd[2],
                   'majAx': majorAxis[2], 'minAx': minorAxis[2],
                   'incl': np.degrees(theta[2])}
            }

    # Plotting.
    stereonetPlot = plt.figure(num='Jelinek plot summary')
    plt.ioff()
    markers = ['s', '^', 'o']
    labels = ['$k_1 = ' + str(round(kK[0], 3)) + 'pm' +
              str(round(kIntervals[0], 3)) + '$',
              '$k_2 = ' + str(round(kK[1], 3)) + 'pm' +
              str(round(kIntervals[1], 3)) + '$',
              '$k_3 = ' + str(round(kK[2], 3)) + '\pm' +
              str(round(kIntervals[2], 3)) + '$']
    ax = stereonetPlot.add_subplot(111, projection='stereonet')
    # Eigenvectors of all tensors
    for tensor in samplePlgTrd:
        for i in range(3):
            ax.line(tensor[0][i], tensor[1][i], markers[i], color='0.3',
                    ms=5, fillstyle='none')
    # Eigenvectors of mean tensor
    for i in range(3):
        ax.line(pKPlg[i], pKTrd[i], markers[i], color='k', ms=7,
                label=labels[i])
    # Confidence regions
    for i in range(3):
        kRegionsLongSplitted, kRegionsLatSplitted = splitIterables(
                kRegionsLong[i], kRegionsLat[i])
        for i in range(len(kRegionsLongSplitted)):
            ax.plot(kRegionsLongSplitted[i], kRegionsLatSplitted[i], ':k',
                    lw=1)
    # Empty plot to add the confidence region legends.
    confLvl = str(round(confLevel * 100, 1))
    ax.line(0, 0, ':k', lw=1,
            label='$'+confLvl + '\%$ confidence regions')
    ax.legend(loc=tuple(np.radians([45, -7])), fontsize='x-small')
    ax.grid(True, ls='--', lw=0.5)
    stereonetPlot.savefig('testForLongSCR.pdf', bbox_inches='tight')
    stereonetPlot.show()

Since it is the same picture than the obtained with the **short script**, it is not displayed again.

