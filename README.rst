===============
``jelinekstat``
===============


.. image:: https://img.shields.io/pypi/v/jelinekstat.svg
        :target: https://pypi.python.org/pypi/jelinekstat

.. image:: https://readthedocs.org/projects/jelinekstat/badge/?version=latest
        :target: https://jelinekstat.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Application software in **Python 3** to apply the statistical proposal of
`Jelínek (1978) <https://doi.org/10.1007/BF01613632>`_ for a sample of several
second-order tensors in order to obtain the mean tensor of the sample, its
principal values with their confidence intervals, and the principal directions
with their confidence regions.

This application program is able to plot the summary of the statistical model
described above in a stereographic projection for a better understanding of the
outcomes. Provided that, the next picture represents the aim of ``jelinekstat``.

.. figure:: https://rawgit.com/eamontoyaa/jelinekstat/master/docs/otherFiles/my_image.svg
        :alt: Outcome plot example

Features
--------

* `Documentation <https://jelinekstat.readthedocs.io>`_
* `PyPI <https://pypi.org/project/jelinekstat>`_
* `GitHub <https://github.com/eamontoyaa/jelinekstat>`_
* Open source and free software: `BSD-2-Clause <https://opensource.org/licenses/BSD-2-Clause>`_.


Requirements
------------

The code was written in Python 3. The packages `numpy <http://www.numpy.org/>`_,
`scipy <https://www.scipy.org/>`_, `matplotlib <https://matplotlib.org/>`_
and `mplstereonet <https://pypi.python.org/pypi/mplstereonet>`_ are
required for using ``jelinekstat``. All of them are
downloadable from the PyPI repository by opening a terminal and typing the
following code lines:


::

    pip install numpy
    pip install scipy
    pip install matplotlib
    pip install mplstereonet


Installation
------------


To install ``jelinekstat`` open a terminal and type:

::

    pip install jelinekstat


Example
-------

To produce the plot shown above execute the following script

::

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


References
----------
Jelínek, V (1978). Statistical processing of anisotropy of magnetic
susceptibility measured on group of specimens. Studia Geophysica et Geodaetica,
22 (1), pp. 50-62.




