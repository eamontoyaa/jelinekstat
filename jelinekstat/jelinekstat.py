'''
Module which contains the required functions to apply the statistical model for
a sample of :math:`n` second-order tensors
`Jelínek (1978) <https://doi.org/10.1007/BF01613632>`_ in order to obtain the
mean tensor :math:`\\boldsymbol{k}` of the sample, the
:math:`\\boldsymbol{k}`'s principal values :math:`k_1, k_2\ \\&\ k_3`, with
their confidence intervals, and the :math:`\\boldsymbol{k}`'s principal
directions :math:`\\boldsymbol{p}_1, \\boldsymbol{p}_2\ \&\ \\boldsymbol{p}_3`
with their confidence regions.

This application program is able to plot the summary of the statistical model
described above in a stereographic projection for a better understanding of the
outcomes.

Note:
    * The packages `numpy <http://www.numpy.org/>`_,\
        `scipy <https://www.scipy.org/>`_,\
        `matplotlib <https://matplotlib.org/>`_\
        and `mplstereonet <https://pypi.python.org/pypi/mplstereonet>`_ are\
        required for using the ``jelinekstat.py`` module. All of them are\
        downloadable from the PyPI repository.
    * The mathematical notation in this documentation is taken from the\
        original reference :cite:`Jelinek1978.article`.
    * Copyright (c) 2018, Universidad Nacional de Colombia, Medellín. \
        Copyright (c) 2018, Exneyder A. Monotoya-Araque and Ludger O. \
        Suarez-Burgoa.\
        `BSD-2-Clause <https://opensource.org/licenses/BSD-2-Clause>`_ or\
        higher.
'''


def normalizeTensors(sample):
    '''Divides all the tensor's elements by the mean susceptibility
    :math:`{k}`, *i.e.* gets :math:`\\boldsymbol{k}_\\mathrm{norm}` by using
    the equations (8) of :cite:`Jelinek1978.article`.

    Parameters:
        sample (`numpy.ndarray`): :math:`(n \\times 6)` array that \
            contains the values obtained from the ``extractdata`` fuction.

    Returns:
        (`numpy.ndarray`): :math:`(n \\times 6)` array that contains the\
            tensors :math:`\\boldsymbol{k}_\\mathrm{norm}` with the same\
            format and structure of the ``extractdata`` function's output.

    Examples:
        >>> from jelinekstat.tools import dataFromFile
        >>> sample, numTensors = dataFromFile('inputDataExample.txt')
        >>> normalizeTensors(sample)
        array(
            [[ 1.02327,  1.02946,  0.94727, -0.01495, -0.03599, -0.05574],
             [1.02315,  1.01803,  0.95882, -0.00924, -0.02058, -0.03151],
             [1.02801,  1.03572,  0.93627, -0.03029, -0.03491, -0.06088],
             [1.02775343,  1.00633335,  0.96591322, -0.01635005, -0.04148014,
              -0.02006007],
             [1.02143,  1.01775,  0.96082, -0.02798, -0.04727, -0.02384],
             [1.01822661,  1.01202663,  0.96974677, -0.01125996, -0.02832991,
              -0.03648988],
             [1.01486338,  1.0206734 ,  0.96446321, -0.01046003, -0.01913006,
              -0.03864013],
             [1.04596,  1.01133,  0.94271, -0.0166 , -0.04711, -0.03636]])
    '''
    import numpy as np

    # Knowing the number of rows
    sample = np.array(sample)
    numTensors = int(np.shape(sample)[0])

    # Normalizing the tensors with the first invariant.
    sampleNorm = np.zeros((0, 6))
    for k in range(0, numTensors):
        a = sample[[k], :] / sum(sample[[k], 0:3][0])
        b = np.array(list(a[0])) * 3
        sampleNorm = np.vstack([sampleNorm, b])

    return sampleNorm


def meantensor(sample, normalized=False):
    '''Estimates the mean tensor :math:`\\boldsymbol{k}` of a randomly chosen
    sample of :math:`n` specimens by using the equation (11) of
    :cite:`Jelinek1978.article` after being normalized the specimens through
    the ``normalizeTensors`` function.

    Parameters:
        sample (`numpy.ndarray`): :math:`(n \\times 6)` array that \
            contains the values of the tensors after being imported with the\
            ``extractdata`` function.
        normalize (`bool`): Logical variable to indicate if the tensors in\
            the ``sample`` variable are already normalized by using the\
            equation (11) of (:cite:`Jelinek1978.article`). ``False`` is the\
            default value. In the case they are not normalized, they will be.

    Returns:
        Three elements are returned; they are described below.

            - **meanTensorVect** (`numpy.ndarray`): Mean tensor in vector\
                form.
            - **meanTensorMtx** (`numpy.ndarray`): Mean tensor in matrix\
                form.
            - **numTensors** (`int`): Number of tensors.

    Examples:
        >>> from jelinekstat.tools import dataFromFile
        >>> from jelinekstat.jelinekstat import meantensor
        >>> sample, numTensors = dataFromFile('inputDataExample.txt')
        >>> meanTensorVect, meanTensorMtx, numTensors = meantensor(
                sample, normalized=False)
        >>> meanTensorVect
        array([ 1.02533293,  1.01891542,  0.95575165, -0.01714126, -0.03435001,
        >>>    -0.03794001])
        >>> meanTensorMtx
        array([[ 1.02533293, -0.01714126, -0.03794001],
               [-0.01714126,  1.01891542, -0.03435001],
               [-0.03794001, -0.03435001,  0.95575165]])
        >>> numTensors
        8
    '''
    import numpy as np
    from jelinekstat.tools import tensorvect2matrixform
    sample = sample.T
    numTensors = np.shape(sample)[1]
    tensorsList = list()
    normalizedTensorsList = list()
    for i in range(numTensors):
        T = sample[:, i]
        tensorsList.append(T)
        if normalized:
            normalizedTensorsList.append(T)
        else:  # normalization with respect first invariant
            inv1 = sum(T[0:3])/3
            normalizedTensorsList.append(T/inv1)
    meanTensorVect = np.mean(normalizedTensorsList, axis=0)
    # mean tensor matricial form
    meanTensorMtx = tensorvect2matrixform(meanTensorVect)
    return meanTensorVect, meanTensorMtx, numTensors


def covMtx2PPlane(
        covMtx, meanTensor, numTensors, tensorVectForm=True):
    '''Obtains the covariance matrix
    :math:`\\boldsymbol{\\mathrm{V^\\mathrm{P}}}` of the
    :math:`\\boldsymbol{k^\\mathrm{P}}`'s elements (*i.e.* going over to a
    Cartesian system determined by
    :math:`\\boldsymbol{p}_1, \\boldsymbol{p}_2\ \&\ \\boldsymbol{p}_3` as is
    shown equation (19) of :cite:`Jelinek1978.article`), by using the equation
    (20) and (21) of :cite:`Jelinek1978.article`.

    Parameters:
        covMtx (`numpy.ndarray`): :math:`(6 \\times 6)` array that estimates\
            the unbiased covariance matrix :math:`\\boldsymbol{\\mathrm{V}}`\
            of the tensors in the sample. It can be obtained by using the\
            equation (13) of :cite:`Jelinek1978.article`) or the ``cov``\
            numpy function (as is shown in the example).
        meanTensor (`numpy.ndarray`): mean tensor :math:`\\boldsymbol{k}` of\
            he sample either in vector or matrix form.
        numTensors (`int`): Number of tensors in the sample.
        tensorVectForm (`bool`): Logical variable to indicate if the input \
            mean tensor is in vector form. ``True`` is the default value.

    Returns:
        (`numpy.ndarray`): :math:`(6 \\times 6)` covariance matrix \
            :math:`\\boldsymbol{\\mathrm{V^\\mathrm{P}}}` of \
            :math:`\\boldsymbol{k^\\mathrm{P}}`'s elements.

    Examples:
        >>> from numpy import cov
        >>> from jelinekstat.tools import dataFromFile
        >>> from jelinekstat.jelinekstat import meantensor, covMtx2PPlane
        >>> sample, numTensors = dataFromFile('inputDataExample.txt')
        >>> normTensors = normalizeTensors(sample)
        >>> meanTensorVect, meanTensorMtx, numTensors = meantensor(
        >>>     normTensors, normalized=True)
        >>> covMtx = cov(normTensors.T, bias=False)
        >>> covMtx2PPlane(
        >>>     covMtx, meanTensorVect, numTensors)
        array([[  1.02065332e-04,   4.74951816e-05,  -1.49560514e-04,
                 -1.65998575e-06,   3.81988803e-05,  -2.24249042e-05],
               [  4.74951816e-05,   5.55684188e-05,  -1.03063600e-04,
                 -2.03320518e-05,  -5.50208910e-06,  -1.06377394e-05],
               [ -1.49560514e-04,  -1.03063600e-04,   2.52624114e-04,
                  2.19920375e-05,  -3.26967912e-05,   3.30626436e-05],
               [ -1.65998575e-06,  -2.03320518e-05,   2.19920375e-05,
                  2.95042484e-05,   2.05384181e-05,  -7.98602106e-06],
               [  3.81988803e-05,  -5.50208910e-06,  -3.26967912e-05,
                  2.05384181e-05,   9.00227092e-05,  -7.40633382e-05],
               [ -2.24249042e-05,  -1.06377394e-05,   3.30626436e-05,
                 -7.98602106e-06,  -7.40633382e-05,   9.59108639e-05]])
    '''
    import numpy as np
    from jelinekstat.tools import tensorvect2matrixform, getEigSorted

    if tensorVectForm:
        meanTensor = tensorvect2matrixform(meanTensor)
    eigVal_mT, eigVec_mT = getEigSorted(meanTensor)
    t = eigVec_mT
    # --- transformation matrix D (equation (21) of Jelinek, 1978) --- #
    D = np.array([[t[0, 0]**2, t[1, 0]**2, t[2, 0]**2,
                   2*t[0, 0]*t[1, 0], 2*t[1, 0]*t[2, 0], 2*t[2, 0]*t[0, 0]],
                  [t[0, 1]**2, t[1, 1]**2, t[2, 1]**2,
                   2*t[0, 1]*t[1, 1], 2*t[1, 1]*t[2, 1], 2*t[2, 1]*t[0, 1]],
                  [t[0, 2]**2, t[1, 2]**2, t[2, 2]**2,
                   2*t[0, 2]*t[1, 2], 2*t[1, 2]*t[2, 2], 2*t[2, 2]*t[0, 2]],
                  [t[0, 0]*t[0, 1], t[1, 0]*t[1, 1], t[2, 0]*t[2, 1],
                   t[0, 0]*t[1, 1]+t[1, 0]*t[0, 1],
                   t[1, 0]*t[2, 1]+t[2, 0]*t[1, 1],
                   t[2, 0]*t[0, 1]+t[0, 0]*t[2, 1]],
                  [t[0, 1]*t[0, 2], t[1, 1]*t[1, 2], t[2, 1]*t[2, 2],
                   t[0, 1]*t[1, 2]+t[1, 1]*t[0, 2],
                   t[1, 1]*t[2, 2]+t[2, 1]*t[1, 2],
                   t[2, 1]*t[0, 2]+t[0, 1]*t[2, 2]],
                  [t[0, 2]*t[0, 0], t[1, 2]*t[1, 0], t[2, 2]*t[2, 0],
                   t[0, 2]*t[1, 0]+t[1, 2]*t[0, 0],
                   t[1, 2]*t[2, 0]+t[2, 2]*t[1, 0],
                   t[2, 2]*t[0, 0]+t[0, 2]*t[2, 0]]
                  ])
    # Here D should be multiplied by (N-1)/N, being N the number of total
    # tensor data but the difference is despreciable for big N, ie for N >= 10.
    D = (numTensors - 1) / numTensors * D
    pCovMtx = np.dot(D, np.dot(covMtx, D.T))  # equation (20) of Jelinek (1978)
    return pCovMtx


def localCovMtxs(meanTensor, pCovMtx, tensorVectForm=True):
    '''Determines the covariance matrix :math:`\\boldsymbol{\\mathrm{W}_i}` of
    the random variables :math:`\\left(\\mathrm{d}p_{ji},
    \\mathrm{d}p_{ki}\\right)` from the local Cartesian System
    :math:`\\mathrm{d}\\boldsymbol{p}_i` that define the :math:`\\mathscr{P}-`
    plane where each confidence area of the mean tensor's princial direcions
    are drawn by using the equation (27) of :cite:`Jelinek1978.article`.

    Parameters:
        meanTensor (`numpy.ndarray`): mean tensor :math:`\\boldsymbol{k}` of\
            the sample either in vector or matrix form.
        pCovMtx (`numpy.ndarray`): :math:`(6 \\times 6)` covariance matrix \
            :math:`\\boldsymbol{\\mathrm{V^\\mathrm{P}}}` of \
            :math:`\\boldsymbol{k^\\mathrm{P}}`'s elements. It is obtained\
            with the ``covMtx2PPlane`` function.
        tensorVectForm (`bool`): Logical variable to indicate if the input \
            :math:`\\boldsymbol{k}` is in vector form. ``True`` is the default\
            value.

    Returns:
        Three elements are returned; they are described below.

            - **W** (`list`): list of three :math:`(2 \\times 2)` arrayes\
                with the covariance matrix :math:`\\boldsymbol{\\mathrm{W}_i}`\
                described above.
            - **eigVal_W** (`list`): list with the three couples of \
                :math:`\\boldsymbol{\\mathrm{W}_i}`'s eigenvalues obtained \
                with the ``getEigSorted`` function.
            - **eigVec_W** (`list`): list with the three :math:`(2 \\times 2)`\
                arrayes of :math:`\\boldsymbol{\\mathrm{W}_i}`'s eigevectors\
                obtained with the ``getEigSorted`` function.

    Examples:
        >>> from numpy import cov
        >>> from jelinekstat.tools import dataFromFile
        >>> from jelinekstat.jelinekstat import meantensor, covMtx2PPlane
        >>> sample, numTensors = dataFromFile('inputDataExample.txt')
        >>> normTensors = normalizeTensors(sample)
        >>> meanTensorVect, meanTensorMtx, numTensors = meantensor(
        >>>     normTensors, normalized=True)
        >>> covMtx = cov(normTensors.T, bias=False)
        >>> pCovMtx = covMtx2PPlane(
        >>>     covMtx, meanTensorVect, numTensors)
        >>> W, eigVal_W, eigVec_W = localCovMtxs(
        >>>     meanTensorVect, pCovMtx)
        >>> W
        [array([[ 0.41635005, -0.00798796],
                [-0.00798796,  0.00679994]]),
         array([[ 0.00739345, -0.02211065],
                [-0.02211065,  0.41635005]]),
        array([[ 0.00679994, -0.00565157],
               [-0.00565157,  0.00739345]])]
        >>> eigVal_W
        [array([ 0.41650579,  0.0066442 ]),
         array([ 0.41754201,  0.00620149]),
         array([ 0.01275605,  0.00143733])]
        >>> eigVec_W
        [array([[ 0.99980999,  0.01949312],
                [-0.01949312,  0.99980999]]),
         array([[ 0.05383071, -0.99855008],
                [-0.99855008, -0.05383071]]),
         array([[ 0.68831829, -0.7254088 ],
                [-0.7254088 , -0.68831829]])]
    '''
    import numpy as np
    from jelinekstat.tools import tensorvect2matrixform, getEigSorted

    if tensorVectForm:
        meanTensor = tensorvect2matrixform(meanTensor)
    eigVal_mT, eigVec_mT = getEigSorted(meanTensor)
    t = eigVal_mT
    W = list()
    eigVal_W = list()
    eigVec_W = list()
    for l, r, s in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        Wl = np.array(
                [[pCovMtx[3 + l, 3 + l] / (t[l] - t[r])**2,
                  pCovMtx[3 + l, 3 + s] / ((t[l] - t[r]) * (t[l] - t[s]))],
                 [pCovMtx[3 + l, 3 + s] / ((t[l] - t[r]) * (t[l] - t[s])),
                  pCovMtx[3 + s, 3 + s] / (t[l] - t[s])**2]])
        W.append(Wl)
        eigVal, eigVec = getEigSorted(Wl)
        eigVal_W.append(eigVal)
        eigVec_W.append(eigVec)
    return W, eigVal_W, eigVec_W


def eigValsIntervals(pCovMtx, numTensors, confLvl=0.95, estimate=True):
    '''Determines the limits of the variabilities of :math:`\\boldsymbol{k}`'s
    principal values for a confidence level given. Ther are obtained by using
    the equation (29) of :cite:`Jelinek1978.article` or their estimate values
    by using the equation (35) of :cite:`Jelinek1978.article`.

    Parameters:
        pCovMtx (`numpy.ndarray`): :math:`(6 \\times 6)` covariance matrix \
            :math:`\\boldsymbol{\\mathrm{V^\\mathrm{P}}}` of \
            :math:`\\boldsymbol{k^\\mathrm{P}}`'s elements. It is obtained\
            with the ``covMtx2PPlane`` function.
        numTensors (`int`): Number of tensors in the sample.
        confLvl (`float`): Confidence level of the limits of the
            variabilities of :math:`\\boldsymbol{k}`'s principal values. \
            ``0.95``  is the default value.
        estimate (`bool`): Logical variable to indicate if the output is based\
            whether on the real or estimate covariance matrix \
            :math:`\\boldsymbol{\\mathrm{V^\\mathrm{P}}}`. ``True`` is the\
            default value.

    Returns:
        (`numpy.ndarray`): Array with the three limits of the variabilities of\
            :math:`\\boldsymbol{k}`'s principal values.

    Examples:
        >>> from numpy import cov
        >>> from jelinekstat.tools import dataFromFile
        >>> from jelinekstat.jelinekstat import *
        >>> sample, numTensors = dataFromFile('inputDataExample.txt')
        >>> normTensors = normalizeTensors(sample)
        >>> meanTensorVect, meanTensorMtx, numTensors = meantensor(
        >>>     normTensors, normalized=True)
        >>> covMtx = cov(normTensors.T, bias=False)
        >>> pCovMtx = covMtx2PPlane(
        >>>     covMtx, meanTensorVect, numTensors)
        >>> eigValsIntervals(
        >>>     pCovMtx, numTensors, confLvl=0.95, estimate=True)
        array([ 0.0084461 ,  0.00623205,  0.01328784])
    '''
    import numpy as np
    from scipy.stats import norm, t

    if estimate:  # t-student distribution
        stat = abs(t.ppf((1-confLvl)/2, df=numTensors-1))  # Eq. (35)
    else:  # normal/gaussian distribution
        stat = abs(norm.ppf((1-confLvl)/2))  # Eq. (29)
    intervals = list()
    variance = np.diag(pCovMtx)/numTensors  # Eq. (26)
    for i in range(3):
        intervals.append(stat * variance[i]**0.5)
    return np.array(intervals)


def eigVectsRegions(W, eigVal_W, eigVec_W, numTensors, confLvl=0.95,
                    estimate=True):
    '''Determines the ellipses' geometric parameters of the confidence regions
    that define the limits of the variabilities of the
    :math:`\\boldsymbol{k}`'s principal vectors.

    The axes lenghts ara obtained by using the equation (32) of
    :cite:`Jelinek1978.article` or their estimated values by
    using the equation (35) of :cite:`Jelinek1978.article` and the inclination
    angles by using the equation (41) of :cite:`Jelinek1978.article`.

    Parameters:
        W (`list`): list of three :math:`(2 \\times 2)` arrayes with the\
            covariance matrix :math:`\\boldsymbol{\\mathrm{W}_i}` described\
            above. It is obtained from the ``eigValsIntervals``\
            function.
        eigVal_W (`list`): list with the three couples of \
            :math:`\\boldsymbol{\\mathrm{W}_i}`'s eigenvalues obtained with \
            the ``getEigSorted`` function. It is obtained from the \
            ``eigValsIntervals`` function.
        eigVec_W (`list`): list with the three :math:`(2 \\times 2)` arrayes \
            of :math:`\\boldsymbol{\\mathrm{W}_i}`'s eigevectors obtained with\
            the ``getEigSorted`` function. It is obtained from the \
            ``eigValsIntervals`` function.
        numTensors (`int`): Number of tensors in the sample.
        confLvl (`float`): Confidence level of the limits of the
            variabilities of :math:`\\boldsymbol{k}`'s principal vectors. \
            ``0.95`` is the default value.
        estimate (`bool`): Logical variable to indicate if the output is based\
            whether on the real or estimate covariance matrix \
            :math:`\\boldsymbol{\\mathrm{V^\\mathrm{P}}}`. ``True`` is the\
            default value.

    Returns:
        Three elements are returned; they are described below.

            - **majorAxis** (`numpy.ndarray`): Array with the three lengths of\
                the ellipses' major axis that define the confidence region. \
                The order is acording to the principal values returned from\
                the ``getEigSorted`` function.
            - **minorAxis** (`list`): list with the three couples of \
                :math:`\\boldsymbol{\\mathrm{W}_i}`'s eigenvalues obtained\
                with the ``getEigSorted`` function. The order is acording to \
                the principal values returned from the ``getEigSorted`` \
                function.
            - **theta** (`list`): list with the three ellipse inclinations in\
                radians measured from the horizontal axis of the local\
                Cartesian System of each ellipse to the respective major axis\
                counter clockwise.

    Examples: ::

        >>> from numpy import cov
        >>> from jelinekstat.tools import dataFromFile
        >>> from jelinekstat.jelinekstat import *
        >>> sample, numTensors = dataFromFile('inputDataExample.txt')
        >>> normTensors = normalizeTensors(sample)
        >>> meanTensorVect, meanTensorMtx, numTensors = meantensor(
        >>>     normTensors, normalized=True)
        >>> covMtx = cov(normTensors.T, bias=False)
        >>> pCovMtx = covMtx2PPlane(
        >>>     covMtx, meanTensorVect, numTensors)
        >>> W, eigVal_W, eigVec_W = localCovMtxs(
        >>>     meanTensorVect, pCovMtx)
        >>> majorAxis, minorAxis, theta = eigVectsRegions(
        >>>     W, eigVal_W, eigVec_W, numTensors, confLvl=0.95,
        >>>     estimate=True)
        >>> majorAxis
        array([ 0.66888885,  0.66949335,  0.13745895])
        >>> minorAxis
        array([ 0.09950548,  0.09615434,  0.04640122])
        >>> theta
        array([-0.01949436, -1.51693959, -0.81162812])
    '''
    import numpy as np
    from scipy.stats import chi2, f

    majorAxis = list()
    minorAxis = list()
    theta = list()
    if estimate:
        stat = f.ppf(confLvl, 2, numTensors-2)  # F distribution
        T2 = stat * 2 * (numTensors-1) / (numTensors * (numTensors-2))
        for i in range(3):  # Eq. (37)
            majorAxis.append(np.arctan((T2 * eigVal_W[i][0])**0.5))
            minorAxis.append(np.arctan((T2 * eigVal_W[i][1])**0.5))
            theta.append(np.arctan(eigVec_W[i][1, 0]/eigVec_W[i][0, 0]))
    else:
        stat = chi2.ppf(confLvl, df=2)  # Chi-square distribution
        for i in range(3):  # Eq. (32)
            majorAxis.append((stat * eigVal_W[i][0] / numTensors)**0.5)
            minorAxis.append((stat * eigVal_W[i][1] / numTensors)**0.5)
            theta.append(np.arctan(eigVec_W[i][1, 0]/eigVec_W[i][0, 0]))
    return np.array(majorAxis), np.array(minorAxis), np.array(theta)


def tensorStat(sample, confLevel=0.95, want2plot=True, plotName='001',
                 ext='pdf'):
    '''Summarizes the :cite:`Jelinek1978.article` statisctic proposal for
    2nd-order tensors and plots it if is wanted.

    Parameters:
        sample (`numpy.ndarray`): :math:`(n \times 6)` array that \
            contains the values obtained from the ``extractdata`` fuction.
        confLvl (`float`): Confidence level of the limits of the
            variabilities of :math:`\boldsymbol{k}`'s principal vectors and \
            values. ``0.95`` is the default value.
        want2plot (`bool`): Logical variable to indicate if is wanted to plot\
            the summary. ``True`` is the default value.
        plotName (`str`): Sample name for saving the final plot. '01' is the\
            default value.
        ext (`str`): File extension for saving the final plot. 'pdf' is the\
            default value.

    Returns:
        (`dict`): Summary of the :cite:`Jelinek1978.article` statisctic\
            proposal for 2nd-order tensors where is stored the data related to\
            the mean tensor and its variability expressed as the variability\
            of their principal values and vectors.

    Examples:
        >>> from jelinekstat.tools import dataFromFile
        >>> from jelinekstat.jelinekstat import tensorStat
        >>> sample, numTensors = dataFromFile('inputDataExample.txt')
        >>> jelinekStatsSummary, stereonetPlot = tensorStat(
        >>>     sample, confLevel=0.95, want2plot=True, plotName='test',
        >>>     ext='pdf')
        >>> jelinekStatsSummary
        {'K': array([[ 1.02533293, -0.01714126, -0.03794001],
                     [-0.01714126,  1.01891542, -0.03435001],
                     [-0.03794001, -0.03435001,  0.95575165]]),
         'k': array([1.02533, 1.01892, 0.95575, -0.01714, -0.03435, -0.03794]),
         'k1': {'mean': 1.0424, 'variability': 0.0084},
         'k2': {'mean': 1.0330, 'variability': 0.0062},
         'k3': {'mean': 0.9236, 'variability': 0.0133},
         'n': 8,
         'p1': {'coords': array([-0.92438729, 0.196842, 0.32674357]),
                'incl': -1.1160, 'majAx': 0.6689, 'minAx': 0.0995,
                'plg': 19.0712, 'trd': 167.9788},
         'p2': {'coords': array([-0.04467227, -0.90653975,  0.41975002]),
                'incl': -86.9142, 'majAx': 0.6695, 'minAx': 0.0962,
                'plg': 24.8188, 'trd': 267.1789},
         'p3': {'coords': array([-0.37883047, -0.37341521, -0.8467872 ]),
                'incl': -46.5029, 'majAx': 0.1375, 'minAx': 0.0464,
                'plg': 57.8639, 'trd': 44.5875}}
        >>> stereonetPlot.show()

         .. plot::

            from jelinekstat.tools import dataFromFile
            from jelinekstat.jelinekstat import tensorStat
            sample, numTensors = dataFromFile('exampledata.txt')
            jelinekStatsSummary, stereonetPlot = tensorStat(
                    sample, confLevel=0.95, want2plot=True, plotName='test',
                    ext='pdf')
            stereonetPlot.show()
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from jelinekstat.tools import getEigSorted, confRegions2PPlanes, \
        eigVects2PlgTrd, proyAllEllipses2LongLat, splitIterables

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
            majorAxis, minorAxis, theta, want2plot, confLevel)

    # Stereographic notation to plot the mean tensor's principal vectors (pK).
    pKPlg, pKTrd = eigVects2PlgTrd(k)  # Plg (plunge); Trd (trend)

    # (plunge,trend) notation to plot principal axis of all tensors.
    samplePlgTrd = list(map(eigVects2PlgTrd, sample))

    # (lon, lat) notation of each confidence region.
    kRegionsLong, kRegionsLat = proyAllEllipses2LongLat(x, y, k)

    # Summary of the Jelinek (1978) statistic proposal for 2nd-order tensors.
    jelinekStatSummary = {
            'K': K,
            'k': k,
            'n': n,
            'k1': {'value': kK[0], 'variability': kIntervals[0]},
            'k2': {'value': kK[1], 'variability': kIntervals[1]},
            'k3': {'value': kK[2], 'variability': kIntervals[2]},
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
    fig = plt.figure(num='Jelinek plot summary')
    if want2plot:
        plt.ioff()
        markers = ['s', '^', 'o']
        labels = ['$k_1 = ' + str(round(kK[0], 3)) + '\pm' +
                  str(round(kIntervals[0], 3)) + '$',
                  '$k_2 = ' + str(round(kK[1], 3)) + '\pm' +
                  str(round(kIntervals[1], 3)) + '$',
                  '$k_3 = ' + str(round(kK[2], 3)) + '\\pm' +
                  str(round(kIntervals[2], 3)) + '$']
        ax = fig.add_subplot(111, projection='stereonet')
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
        ax.line(0, 0, ':k', lw=1, label='$'+confLvl + '\%$ conf. regions')
        ax.legend(loc=tuple(np.radians([45, -7])), fontsize='x-small')
        ax.grid(True, ls='--', lw=0.5)
        fig.savefig(plotName + '.' + ext, bbox_inches='tight')

    return jelinekStatSummary, fig
