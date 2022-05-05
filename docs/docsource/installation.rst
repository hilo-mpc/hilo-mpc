===============
Installation
===============
HILO-MPC is distributed as a Python package. We recommend to create a new Python environment and install HILO-MPC and
the other necessary packages there. `Here <https://docs.python.org/3/tutorial/venv.html>`_ you can find the
instructions on how to create a virtual environment using venv.

Installation from PyPI
======================
After activating your Python environment, run the following command in your terminal

.. code-block::

    pip install hilo-mpc

Hard dependencies, that is necessary packages (like CasADi or NumPy), will be installed automatically.

Clone from GitHub
=================
You can also clone the files directly from GitHub running

.. code-block::

    git clone https://github.com/hilo-mpc/hilo-mpc.git

Additional packages
===================
HILO-MPC can make use of a few Python libraries that are not automatically installed since you might not need all of
them. If you need them for you application, please install them manually.

.. note::

    Make sure you install the correct version of these libraries

    ============   ===============
    Library        Version
    ============   ===============
    TensorFlow     >=2.3.0, <2.8.0
    PyTorch        >=1.2.0
    scikit-learn   >=0.19.2
    Bokeh          >=2.3.0
    Matplotlib     >=3.0.0
    pandas         >=1.0.0
    ============   ===============