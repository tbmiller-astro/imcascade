Installation
============

The source code for ``imcascade`` is stored in the github repo: https://github.com/tbmiller-astro/imcascade

To install ``imcascade`` simply clone the github repo and run the setup.py install script

.. code-block:: bash

  $ cd < Directory where imcascade will be installed >
  $ git clone https://github.com/tbmiller-astro/imcascade
  $ cd imcascade
  $ python setup.py install

We are planning to upload ``imcascade`` to PyPI for even easier installation using ``pip`` so check back soon!

Dependencies
------------
``imcascade`` is written purely in Python and requires the following packages, all of which can be installed using either ``pip install`` or ``conda install``

* ``numpy``

* ``scipy``

* ``matplotlib``

* ``astropy``

* ``numba``

* ``sep``

* ``dyensty``

* ``asdf``

``imcascade`` was developed and tested using Python 3, but it may also work in Python 2 if the required packages can be installed, but be careful that it has not been properly vetted.
