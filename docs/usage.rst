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

The first way is by using the function ``tensorStat`` from the ``jelinekstat.py`` module in a short script as follow.

.. literalinclude:: ../examples/shortSCR.py
   :language: python

.. figure:: https://rawgit.com/eamontoyaa/jelinekstat/master/examples/figures/jelinekstat_tensorStat_example.svg
    :alt: shortSCR_example

.. only:: html

   :download:`example script<../examples/shortSCR.py>`.



Long script
^^^^^^^^^^^

The second way is by using all the code lines inside the same function used above in a much longer script as follow.


.. literalinclude:: ../examples/longSCR.py
   :language: python

.. only:: html

   :download:`example script<../examples/longSCR.py>`.


Since it is the same picture than the obtained with the **short script**, it is not displayed again.

