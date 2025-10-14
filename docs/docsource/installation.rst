===============
Installation
===============
HILO-MPC is distributed as a Python package with support for **Python 3.9–3.12**. We recommend creating a new Python environment and installing HILO-MPC there. `Here <https://docs.python.org/3/tutorial/venv.html>`_ you can find instructions on how to create a virtual environment using venv.

Installation with Poetry (Recommended)
=======================================
For development or when you want fine control over dependencies:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/hilo-mpc/hilo-mpc.git
    cd hilo-mpc
    
    # Install core package
    poetry install
    
    # Install with optional extras
    poetry install -E ml -E viz -E tensorflow

Installation from PyPI
======================
After activating your Python environment, run:

.. code-block:: bash

    # Core installation
    pip install hilo-mpc
    
    # With optional extras
    pip install hilo-mpc[ml,viz,tensorflow]

Core dependencies (CasADi, NumPy, SciPy) are installed automatically.

Optional Dependencies
=====================
HILO-MPC uses a minimal core installation by default. Additional features require optional dependencies, which are kept optional to avoid forcing users to install heavy packages they may not need.

Install Optional Extras
-----------------------

**With Poetry:**

.. code-block:: bash

    # Machine learning utilities (scikit-learn, pandas)
    poetry install -E ml
    
    # Plotting backends (Bokeh, Matplotlib)
    poetry install -E viz
    
    # TensorFlow backend for neural networks
    poetry install -E tensorflow
    
    # PyTorch backend for neural networks
    poetry install -E pytorch
    
    # Install multiple extras
    poetry install -E ml -E viz -E tensorflow

**With pip:**

.. code-block:: bash

    pip install hilo-mpc[ml,viz,tensorflow,pytorch]

Optional Package Versions
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 15 45

   * - Extra
     - Packages
     - Version
     - Purpose
   * - ``ml``
     - scikit-learn
     - ≥0.19.2
     - Data preprocessing and normalization
   * - 
     - pandas
     - ≥1.0.0, ≤1.5.1
     - Data handling for training
   * - ``viz``
     - Bokeh
     - ≥2.3.0
     - Interactive plotting
   * - 
     - Matplotlib
     - ≥3.0.0
     - Static plotting
   * - ``tensorflow``
     - TensorFlow
     - ≥2.8.0
     - Neural network training (TensorFlow backend)
   * - 
     - TensorBoard
     - ≥2.8.0
     - Training visualization
   * - ``pytorch``
     - PyTorch
     - ≥1.2.0
     - Neural network training (PyTorch backend)
   * - 
     - TorchVision
     - ≥0.4.0
     - PyTorch utilities

.. note::

    The package will raise informative errors if you try to use features that require uninstalled optional dependencies. This design keeps the core installation lightweight while allowing users to install only what they need.

Clone from GitHub
=================
You can also clone the repository directly:

.. code-block:: bash

    git clone https://github.com/hilo-mpc/hilo-mpc.git