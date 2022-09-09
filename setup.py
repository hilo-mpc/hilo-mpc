#!/usr/bin/env python3

import os
from setuptools import setup, find_packages


def read(file_name):
    """
    Utility function to read the README file. Used for long_description.

    :param file_name:
    :return:
    """
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


setup(
    name='HILO-MPC',
    version='1.0.2',
    description='HILO-MPC is a toolbox for easy, flexible and fast development of machine-learning-supported optimal '
                'control and estimation problems',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    author='',
    author_email='',
    maintainer='HILO-MPC Developers',
    maintainer_email='',
    url='https://www.ccps.tu-darmstadt.de/research_ccps/hilo_mpc/',
    download_url='https://github.com/hilo-mpc/hilo-mpc/releases',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    license='GNU Lesser General Public License v3 (LGPLv3)',
    platforms=['Windows', 'Linux'],
    install_requires=[
        'casadi>=3.5',
        'numpy<=1.19.5',
        'scipy',
        'prettytable'
    ],
    python_requires='>=3.7',
    project_urls={
        'Bug Tracker': 'https://github.com/hilo-mpc/hilo-mpc/issues',
        'Documentation': 'https://hilo-mpc.github.io/hilo-mpc/',
        'Source Code': 'https://github.com/hilo-mpc/hilo-mpc'
    }
)
