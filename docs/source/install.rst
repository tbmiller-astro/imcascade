Installation
============

The source code for ``imcascade`` is stored on `github <https://github.com/tbmiller-astro/imcascade>`_

To install ``imcascade`` simply clone the github repo and run the setup.py install script. This is the best way to make sure you are using the most up to date version.

.. code-block:: bash

  $ cd < Directory where imcascade will be installed >
  $ git clone https://github.com/tbmiller-astro/imcascade
  $ cd imcascade
  $ python setup.py install

We have also uploaded our code to PyPI so you can install ``imcascade`` with pip

.. code-block:: bash

  $ pip install imcascade


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

Optionally, you can also install ``jax``, this speeds up the express sampling by a decent margin. It is left as optional as it cannot currently be installed easily on all systems. To install jax follow the instructions `here. <https://github.com/google/jax#installation>`_

``imcascade`` was developed and tested using Python 3, but it may also work in Python 2 if the required packages can be installed, but be careful that it has not been properly vetted.
