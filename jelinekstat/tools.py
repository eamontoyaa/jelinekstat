'''
Module which contains the functions that are supportive tools for the functions
contained in the ``jelinekstat.py`` module which is the guideline of the
second-order tensors statistical proposal of
`Jelínek (1978) <https://doi.org/10.1007/BF01613632>`_.

Note:
    * The packages `numpy <http://www.numpy.org/>`_,\
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

References:
    .. bibliography:: references.bib
'''


def dataFromFile(file):
    '''Loads the ``.txt`` file with all second-order tensors components.

    Parameters:
        file (`str`): ``txt`` file tabulated with tabular spaces as delimiter.\
            The file is structured as a :math:`(n \\times 6)` array, where \
            :math:`n` is the number of tensors and each row contains the\
            vector form with the 6 components of a tensor in the following\
            order :math:`t_{11}, t_{22}, t_{33}, t_{12}, t_{23}, t_{13}`.

    Returns:
        Two elements are returned; they are described below.

            - **sample** (`numpy.ndarray`): :math:`(n \\times 6)` array\
                that contains the same values than the ``.txt`` file.
            - **numTensors** (`int`): Number of tensors.

    Examples:
        >>> from jelinekstat.tools import dataFromFile
        >>> sample, numTensors = dataFromFile('inputDataExample.txt')
        >>> sample
        array([[ 1.02327,  1.02946,  0.94727, -0.01495, -0.03599, -0.05574],
               [ 1.02315,  1.01803,  0.95882, -0.00924, -0.02058, -0.03151],
               [ 1.02801,  1.03572,  0.93627, -0.03029, -0.03491, -0.06088],
               [ 1.02775,  1.00633,  0.96591, -0.01635, -0.04148, -0.02006],
               [ 1.02143,  1.01775,  0.96082, -0.02798, -0.04727, -0.02384],
               [ 1.01823,  1.01203,  0.96975, -0.01126, -0.02833, -0.03649],
               [ 1.01486,  1.02067,  0.96446, -0.01046, -0.01913, -0.03864],
               [ 1.04596,  1.01133,  0.94271, -0.0166 , -0.04711, -0.03636]])
        >>> numTensors
        8
    '''
    import numpy as np

    # String file converted to numpy array.
    sample = np.loadtxt(file)
    # Number of tensors in the file.
    numTensors = int(np.shape(sample)[0])

    return sample, numTensors


def tensorvect2matrixform(tensorVect):
    '''Converts a second order tensor from the vector form of to its matricial
    form.

    Parameters:
        tensorVect (`list` or `numpy.ndarray`): :math:`(n \\times 6)` tensor\
            components written as column vector with the following order\
            :math:`t_{11}, t_{22}, t_{33}, t_{12}, t_{23}, t_{13}`.

    Returns:
        (`numpy.ndarray`): :math:`(3 \\times 3)` second-order tensor expressed\
            as a :math:`3\\times 3` matrix.

    Examples:
        >>> from jelinekstat.tools import tensorvect2matrixform
        >>> tensorVect = [11, 22, 33, 12, 23, 13]
        >>> tensorvect2matrixform(tensorVect)
        array([[11, 12, 13],
               [12, 22, 23],
               [13, 23, 33]])
    '''
    import numpy as np

    tensorVect = np.array(tensorVect)
    tensorMtxForm = np.array([[tensorVect[0], tensorVect[3], tensorVect[5]],
                              [tensorVect[3], tensorVect[1], tensorVect[4]],
                              [tensorVect[5], tensorVect[4], tensorVect[2]]])
    return tensorMtxForm


def vector2plungetrend(vector):
    '''Converts a :math:`\\mathbb{R}^3` vector to the **plunge**
    :math:`\\delta`, **trend** :math:`\\delta_\\mathrm{dir}` notation used in
    Structural Geology and Rock Mechanics.

    The :math:`\\mathbb{R}^3` notation is assumed to be coincident with the
    **NED** notation (*i.e* North, East, Nadir).

    Parameters:
        vector (`list` or `numpy.ndarray`): :math:`\\left(x, y, z \\right)`\
            vector.

    Returns:
        (`tuple`): :math:`\\left(\\delta, \\delta_\\mathrm{dir}\\right)` of\
            the input vector in degrees.

    Examples:
        >>> from jelinekstat.tools import vector2plungetrend
        >>> vector = [1, 0, 0]
        >>> vector2plungetrend(vector)
        (0.0, 0.0)

        >>> from jelinekstat.tools import vector2plungetrend
        >>> vector = [0, 1, 0]
        >>> vector2plungetrend(vector)
        (0.0, 90.0)

        >>> from jelinekstat.tools import vector2plungetrend
        >>> vector = [1, 1, 1]
        >>> vector2plungetrend(vector)
        (35.264389682754654, 45.0)
    '''
    import numpy as np

    vector = np.array(vector)
    x = vector[0]
    y = vector[1]
    z = vector[2]
    plunge = np.degrees(np.arctan(z / np.sqrt(x**2 + y**2)))
    trend = np.degrees(np.arctan2(y, x)) % 360
    if plunge < 0:
        trend = (trend + 180) % 360
    return abs(plunge), trend


def getEigSorted(matrix):
    '''Obtains eigenvalues and eigenvectors of a diagonalizable matrix. The
    eigenvalues are sorted descending.

    Parameters:
        matrix (`numpy.ndarray`): :math:`(3 \\times 3)` diagonalizable matrix.

    Returns:
        Two elements are returned; they are described below.

            - **sortedEigVal** (`numpy.ndarray`): :math:`(3 \\times 1)` array\
                with the eigenvalues ordered descending.
            - **sortedEigVec** (`numpy.ndarray`): :math:`(3 \\times 3)` array\
                with the eigenvectors, such that the column \
                ``sortedEigVec[:, i]`` is the eigenvector corresponding to the\
                eigenvalue ``sortedEigVal[i]``
    Examples:
        >>> from numpy import array
        >>> from jelinekstat.tools import getEigSorted
        >>> matrix = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> sortedEigVal, sortedEigVec = getEigSorted(matrix)
        >>> sortedEigVal
        array([  1.61168440e+01,  -9.75918483e-16,  -1.11684397e+00])
        >>> sortedEigVec
        array([[-0.23197069,  0.40824829, -0.78583024],
               [-0.52532209, -0.81649658, -0.08675134],
               [-0.8186735 ,  0.40824829,  0.61232756]])
    '''
    import numpy as np

    eigVal, eigVec = np.linalg.eig(matrix)
    idx = eigVal.argsort()[::-1]
    sortedEigVal = eigVal[idx]
    sortedEigVec = eigVec[:, idx]
    return sortedEigVal, sortedEigVec


def confRegions2PPlanes(majorAxis, minorAxis, theta, want2plot=True,
                        confLvl=0.95):
    '''Determines the :math:`\\mathbb{R}^2` coordinates of each confidence
    ellipse in the local Cartesyan System of the :math:`\\mathscr{P}-` planes
    that contain them. It is donde from their geometric parameters obtained
    from the ``eigVectsRegions`` function. If it is wanted,
    plots them too.

    Parameters:
        majorAxis (`numpy.ndarray`): Array with the three lengths of the \
            ellipses' major axis that define the confidence region. The order \
            is acording to the principal values returned from the \
            ``getEigSorted`` function.
        minorAxis (`list`): list with the three couples of \
            :math:`\\boldsymbol{\\mathrm{W}_i}`'s eigenvalues obtained with\
            the ``getEigSorted`` function. The order is acording to the\
            principal values returned from the ``getEigSorted`` function.
        theta (`list`): list with the three ellipse inclinations in radians \
            measured from the horizontal axis of the local Cartesian System of\
            each ellipse to the respective major axis counter clockwise.
        want2plot (`bool`): Logical variable to indicate if is wanted to plot\
            the ellipes. ``True`` is the default value.
        confLvl (`float`): Confidence level of the limits of the
            variabilities of :math:`\\boldsymbol{k}`'s principal vectors and \
            values. ``0.95`` is the default value.

    Returns:
        Three elements are returned; they are described below.

            - **x** (`list`): List of three arrays that contain the abscises\
                of the each ellipse.
            - **y** (`list`): List of three arrays that contain the ordinates\
                of the each ellipse.
            - **fig** (`list`): ``matplotlib`` object. use ``fig.show()`` for\
                displaying the plot

    Examples:
        >>> from numpy import array
        >>> from jelinekstat.tools import confRegions2PPlanes
        >>> majorAxis = array([ 0.66888885,  0.66949335,  0.13745895])
        >>> minorAxis = array([ 0.09950548,  0.09615434,  0.04640122])
        >>> theta = array([-0.01949436, -1.51693959, -0.81162812])
        >>> x, y, fig = confRegions2PPlanes(majorAxis, minorAxis, theta,
        >>>                                 want2plot=True, confLvl=0.95)
        >>> fig.show()

        .. plot::

            # Full example
            from numpy import cov
            from jelinekstat.tools import dataFromFile, confRegions2PPlanes
            from jelinekstat.jelinekstat import normalizeTensors,\
                meantensor, covMtx2PPlane, localCovMtxs, eigVectsRegions
            sample, numTensors = dataFromFile('exampledata.txt')
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
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    numPoints = 500  # Number of points for each ellipse
    confLvl = round(confLvl * 100, 1)
    phi = np.linspace(0, 2 * np.pi, numPoints)
    x, y = list(), list()
    for i in range(3):
        rotMatrix = np.array([[np.cos(theta[i]), -np.sin(theta[i])],
                              [np.sin(theta[i]), np.cos(theta[i])]])
        rotMatrix = [rotMatrix] * numPoints
        xyNoRot = np.array([np.cos(phi)*majorAxis[i],
                            np.sin(phi)*minorAxis[i]]).T
        xyRot = np.array(list(map(np.dot, rotMatrix, xyNoRot)))
        x.append(xyRot[:, 0])
        y.append(xyRot[:, 1])
        # plot of each ellipse
        plt.ioff()
        fig = plt.figure(num='Confidence regions')
        eigenVectName = ['major', 'intermediate', 'minor']
        xLabel = ['$p_2$', '$p_3$', '$p_1$']
        yLabel = ['$p_3$', '$p_1$', '$p_2$']
        if want2plot:
            ax = fig.add_subplot(1, 3, i+1)
            ax.plot(x[i], y[i], '-k')
            ax.axis('equal')
            ax.grid(True, ls='--', lw=0.5)
            ax.set_xlabel(xLabel[i])
            ax.set_ylabel(yLabel[i])
            ax.set_title(str(confLvl)+'% confidence ellipse \nof ' +
                         eigenVectName[i] + ' eigenvector', fontweight='bold',
                         fontsize=9)
            ax.tick_params(labelsize=7)
            fig.subplots_adjust(wspace=0.7)
    return x, y, fig


def rotateaxis2proyectellipses(axisN, axisE, axisD):
    '''Since it is easier, the projection of an ellpise (confidence region) on
    the stereogrpahic net is thought as a serie of rotations from the bottom of
    the semi-sphere, *i.e.*, the *nadir*.

    This function determines the axes names around which is necesary to rotate
    a confidence ellipse once it is placed at nadir of a semi-sphere to poject
    her from the nadir to the real position on the semi-sphere. Besides, it
    determines the angles to rotate at each axis name.

    The ``mplstereonet`` reference system has the :math:`x, y` and :math:`z`
    vectors as its base, and they correspond to the *nadir*, *east* and *north*
    vectors in the **NED** reference system of the semi-spherical space of the
    Stereographic Projection.

    It is implicit that the three input vectors are orthogonal to each other
    due to they correspond to the principal vectors of :math:`\\boldsymbol{k}`.

    Parameters:
        axisN (`numpy.ndarray`): Array with the coordinates :math:`x, y, z`\
            of the eigenvector that will point to the north-axis once the
            ellipse is placed at nadir of the semi-sphere, *i.e.*, its
            othogonal eigenvector associated points downward.
        axisE (`numpy.ndarray`): Array with the coordinates :math:`x, y, z`\
            of the eigenvector that will point to the east-axis once the
            ellipse is placed at nadir of the semi-sphere, *i.e.*, its
            othogonal eigenvector associated points downward.
        axisD (`numpy.ndarray`): Array with the coordinates :math:`x, y, z`\
            of the eigenvector that is othogonal to the ellipse.

    Returns:
        Two elements are returned; they are described below.

            - **axis2rot** (`list`): Strings with the axis-names of the\
                **NED** system around which will be done the rotatios to\
                project the confidence ellipse.
            - **angles2rot** (`list`): List of the angles in degrees for\
                rotating a ellipse once it is placed orthogonal to the nadir.

    Examples:
        >>> from numpy import array
        >>> from jelinekstat.tools import rotateaxis2proyectellipses
        >>> axis2rot, angles2rot = rotateaxis2proyectellipses(
        >>>     array([2, 2, 1]), array([-2, 1, 2]), array([1, -2, 2]))
        >>> axis2rot
        ['E', 'D', 'N', 'D']
        >>> angles2rot
        [-180, 26.565051177077976, 48.189685104221404, 206.56505117707798]
    '''
    import numpy as np
    from mplstereonet.stereonet_math import geographic2plunge_bearing, line
    from mplstereonet.stereonet_math import _rotate as rot

    # Nadir axis
    axisDplg, axisDtrd = vector2plungetrend(axisD)
    axisDlong, axisDlat = line(axisDplg, axisDtrd)
    # East axis
    axisEplg, axisEtrd = vector2plungetrend(axisE)
    axisElong, axisElat = line(axisEplg, axisEtrd)
    # North axis
    axisNplg, axisNtrd = vector2plungetrend(axisN)
    axisNlong, axisNlat = line(axisNplg, axisNtrd)

    # rotation around the nadir axis to put axisD on the E-W line
    angle1 = 90 - axisDtrd
    rot1long, rot1lat = rot(np.degrees([axisNlong, axisElong, axisDlong]),
                            np.degrees([axisNlat, axisElat, axisDlat]),
                            angle1, axis='x')
    rot1plg, rot1trd = geographic2plunge_bearing(rot1long, rot1lat)

    # rotation around the north axis to put axisD on the nadir axis
    angle2 = np.degrees(rot1long[2][0])
    if rot1long[2] > 0:
        angle2 *= -1
    rot2long, rot2lat = rot(np.degrees(rot1long), np.degrees(rot1lat),
                            angle2, axis='z')
    rot2plg, rot2trd = geographic2plunge_bearing(rot2long, rot2lat)

    # rotation around the nadir axis to put axisE on the east axis
    angle3 = 90 - rot2trd[1][0]
    rot3long, rot3lat = rot(np.degrees(rot2long), np.degrees(rot2lat),
                            angle3, axis='x')
    rot3plg, rot3trd = geographic2plunge_bearing(rot3long, rot3lat)

    # rotation around the east axis to put axisN on the north axis
    if rot3lat[0] < 0:
        angle4 = 180
    else:
        angle4 = 0
    rot4long, rot4lat = rot(np.degrees(rot3long), np.degrees(rot3lat),
                            angle4, axis='y')
    rot4plg, rot4trd = geographic2plunge_bearing(rot4long, rot4lat)

    axis2rot = ['E', 'D', 'N', 'D']  # ['y', 'x', 'z', 'x'] sensu mplstereonet
    angles2rot = [-angle4, -angle3, -angle2, -angle1]
    return axis2rot, angles2rot


def proyAnEllipse2LongLat(x, y, axis2rot, angles2rot):
    '''Pojects just an ellipse from the :math:`\\mathbb{R}^2` coordinates of
    the :math:`\\mathscr{P}-` plane that contains it to the real position on
    the stereographic projection through some rotations from an initial
    position at the nadir of the semi-sphere.

    Parameters:
        x (`numpy.ndarray` or `list`): Abscises of just one ellipse's boundary\
            on the :math:`\\mathscr{P}-` plane. It is obtained from the \
            ``confRegions2PPlanes`` function.
        y (`numpy.ndarray` or `list`): Ordinates of just one ellipse's\
            boundary on the :math:`\\mathscr{P}-` plane. It is obtained from\
            the ``confRegions2PPlanes`` function.
        axis2rot (`list`): Strings with the axis-names of the **NED** system\
            around which will be done the rotatios to project the confidence\
            ellipse. It is obtained from the ``rotateaxis2proyectellipses`` \
            function.
        angles2rot (`list`): List of the angles in degrees for rotating a\
            ellipse once it is placed orthogonal to the nadir. It is obtained\
            from the ``rotateaxis2proyectellipses`` function.

    Returns:
        Two elements are returned; they are described below.

            - **ellipLong** (`numpy.ndarray`): Longitudes of the ellipse's\
                boundary after being rotated to its right position in the\
                stereographic projection.
            - **ellipLat** (`numpy.ndarray`): Latitudes of the ellipse's\
                boundary after being rotated to its right position in the\
                stereographic projection.

    Examples:
        >>> from numpy import array
        >>> from jelinekstat.tools import confRegions2PPlanes
        >>> majorAxis = array([ 0.66888885,  0.66949335,  0.13745895])
        >>> minorAxis = array([ 0.09950548,  0.09615434,  0.04640122])
        >>> theta = array([-0.01949436, -1.51693959, -0.81162812])
        >>> x, y = confRegions2PPlanes(majorAxis, minorAxis, theta, False,
        >>>                            0.95)
        >>> ellipLong, ellipLat = proyAnEllipse2LongLat(
        >>>     x[0], y[0], ['E', 'D', 'N', 'D'], [-180, 116.37, 70.93, 77.98])
    '''
    import numpy as np
    from mplstereonet.stereonet_math import geographic2plunge_bearing, line
    from mplstereonet.stereonet_math import _rotate as rot

    # transform NED to xyz-mplstereonet axes notation
    for i in range(len(axis2rot)):
        if axis2rot[i] == 'N':
            axis2rot[i] = 'z'
        elif axis2rot[i] == 'E':
            axis2rot[i] = 'y'
        elif axis2rot[i] == 'D':
            axis2rot[i] = 'x'
    vectOnes = np.ones(len(x))
    ellipNED = np.array([y, x, vectOnes]).T
    ellipPlgTrd = np.array(list(map(vector2plungetrend, ellipNED)))
    ellipPlg = ellipPlgTrd[:, 0]
    ellipTrd = ellipPlgTrd[:, 1]
    ellipLong, ellipLat = line(ellipPlg, ellipTrd)
    for i in range(len(angles2rot)):
        ellipLong, ellipLat = rot(np.degrees(ellipLong), np.degrees(ellipLat),
                                  angles2rot[i], axis=axis2rot[i])
    ellipPlg, ellipTrd = geographic2plunge_bearing(ellipLong, ellipLat)
    ellipLong, ellipLat = line(ellipPlg, ellipTrd)
    return ellipLong, ellipLat


def eigVects2PlgTrd(tensor, tensorVectForm=True):
    '''Obtains the principal vectors of a second-order tensor and returns them
    in the the **plunge**, **trend** :math:`\\left(\\delta,
    \\delta_\\mathrm{dir}\\right)` notation used in Structural Geology and Rock
    Mechanics.

    Parameters:
        tensor (`numpy.ndarray`): A secon-order tensor.
        tensorVectForm (`bool`): Logical variable to indicate if the input \
            tensor is in vector form. ``True`` is the default value.

    Returns:
        Two elements are returned; they are described below.

            - **eigVecPlg** (`list`): Plunges of the three principal vectors \
                of the input tensor. The order is acording to the principal\
                values returned from the ``getEigSorted`` function.
            - **eigVecTrd** (`list`): Trends of the three principal vectors of\
                the input tensor. The order is acording to the principal\
                values returned from the ``getEigSorted`` function.

    Examples:
        >>> from numpy import array
        >>> from jelinekstat.tools import eigVects2PlgTrd
        >>> tensor = array([1.023, 1.0295,  0.9473, -0.0150, -0.0360, -0.056])
        >>> eigVecPlg, eigVecTrd = eigVects2PlgTrd(
        >>>     tensor, tensorVectForm=True)
        >>> eigVecPlg
        [31.176002127688509, 7.2363353791762837, 57.806971200122995]
        >>> eigVecTrd
        [198.07283722120425, 292.47894708173089, 34.114473250861067]
    '''
    if tensorVectForm:
        tensor = tensorvect2matrixform(tensor)
    eigVal, eigVec = getEigSorted(tensor)
    eigVecPlg = list()
    eigVecTrd = list()
    for i in range(3):
        plg, trd = vector2plungetrend(eigVec[:, i])
        eigVecPlg.append(plg)
        eigVecTrd.append(trd)
    return eigVecPlg, eigVecTrd


def proyAllEllipses2LongLat(x, y, meanTensor, tensorVectForm=True):
    '''Pojects all the three confidence ellipses from the :math:`mathbb{R}^2`
    coordinates of the :math:`mathscr{P}-` plane that contain them to the
    real position on the stereographic projection through some rotations from
    an initial position at the nadir of the semi-sphere.

    Parameters:
        x (`numpy.ndarray` or `list`): Arrangement of the three lists each one\
            with the abscises of an ellipse's boundary on the \
            :math:`mathscr{P}-` plane. They are obtained from the \
            ``confRegions2PPlanes`` function.
        y (`numpy.ndarray` or `list`): Arrangement of the three lists each one\
            with the ordinates of an ellipse's boundary on the \
            :math:`mathscr{P}-` plane. They are obtained from the \
            ``confRegions2PPlanes`` function.
        meanTensor (`numpy.ndarray`): mean tensor :math:`\\boldsymbol{k}` of\
            the sample either in vector or matrix form.
        tensorVectForm (`bool`): Logical variable to indicate if the input \
            :math:`\\boldsymbol{k}` is in vector form. ``True`` is the default\
            value.

    Returns:
        Two elements are returned; they are described below.

            - **long** (`numpy.ndarray`): Array of the three lists each one\
                with the longitudes (in radians) of all the ellipse's boundary\
                after being rotated to its right position in the stereographic\
                projection.
            - **lat** (`numpy.ndarray`): Array of the three lists each one\
                with the latitudes (in radians) of all the ellipse's boundary\
                after being rotated to its right position in the stereographic\
                projection.

    Examples:
        >>> from numpy import array
        >>> from jelinekstat.tools import *
        >>> from jelinekstat.jelinekstat import meantensor
        >>> sample, numTensors = dataFromFile('inputDataExample.txt')
        >>> meanTensorVect, meanTensorMtx, numTensors = meantensor(
        >>>     sample, normalized=True)
        >>> majorAxis = array([ 0.66888885,  0.66949335,  0.13745895])
        >>> minorAxis = array([ 0.09950548,  0.09615434,  0.04640122])
        >>> theta = array([-0.01949436, -1.51693959, -0.81162812])
        >>> x, y = confRegions2PPlanes(
        >>>     majorAxis, minorAxis, theta, False, 0.95)
        >>> long, lat = proyAllEllipses2LongLat(x, y, meanTensorVect)
    '''

    if tensorVectForm:
        meanTensor = tensorvect2matrixform(meanTensor)
    eigVal, eigVec = getEigSorted(meanTensor)
    long = list()
    lat = list()
    for l, r, s in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        axisD = eigVec[:, l]  # Nadir axis
        axisE = eigVec[:, r]  # East axis
        axisN = eigVec[:, s]  # North axis
        axis2rot, angles2rot = rotateaxis2proyectellipses(axisN, axisE, axisD)
        ellipLong, ellipLat = proyAnEllipse2LongLat(x[l], y[l],
                                                    axis2rot, angles2rot)
        long.append(ellipLong)
        lat.append(ellipLat)
    return long, lat


def splitIterables(iter1, iter2):
    '''Splits two iterable elements which are paired by selecting the math:`n`
    common indexes where there are sign changes in both inputs at the same
    time. If there is any index to split the inputs, it returns the same inputs
    within a list.

    Parameters:
        iter1 (`numpy.ndarray` or `list`): An iterable element which is\
            paired to ``iter2``.
        iter2 (`numpy.ndarray` or `list`): An iterable element which is\
            paired to ``iter1``.

    Returns:
        Two elements are returned; they are described below.

            - **iter1Splitted** (`list`): Segments of the original ``iter1``\
                input after being splitted.
            - **iter2splitted** (`list`): Segments of the original ``iter2``\
                input after being splitted.

    Examples:
        >>> from jelinekstat.tools import splitIterables
        >>> iter1, iter2 = [1, -2, -3, 4, 5, 6], [-3, -2, -1, 0, 1, 2]
        >>> iter1Splitted, iter2splitted = splitIterables(iter1, iter2)
        >>> iter1Splitted
        [[1, -2, -3], [4, 5, 6]]
        >>> iter2splitted
        [[-3, -2, -1], [0, 1, 2]]
    '''
    import numpy as np

    signPosit1 = np.where(np.diff(np.sign(iter1)))[0]
    signPosit2 = np.where(np.diff(np.sign(iter2)))[0]
    signPosit = np.intersect1d(signPosit1, signPosit2)
    iter1Splitted = list()
    iter2splitted = list()
    if len(signPosit) == 0:
        iter1Splitted.append(iter1)
        iter2splitted.append(iter2)
    else:
        iter1Splitted.append(iter1[:signPosit[0]+1])
        iter2splitted.append(iter2[:signPosit[0]+1])
        for i in range(len(signPosit)-1):
            iter1Splitted.append(iter1[signPosit[i]+1:signPosit[i+1]])
            iter2splitted.append(iter2[signPosit[i]+1:signPosit[i+1]])
        iter1Splitted.append(iter1[signPosit[-1]+1:])
        iter2splitted.append(iter2[signPosit[-1]+1:])
    return iter1Splitted, iter2splitted
