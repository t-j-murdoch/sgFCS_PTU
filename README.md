# sgFCS_PTU

## Python based shrinking gate fluorescence correlation spectroscopy
This package proccess TCSPC data in the form of a PTU file with both lifetime fitting and shrinking gate Fluorescence spectrscopy.

## Installation
This package has been tested in a conda environment on a Windows machine. A possible installation path is:

-Make sure [appropriate build tools](https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.x_with_Visual_Studio_2022_.28x86.2C_x64.2C_ARM.2C_ARM64.29) are installed
-[Install Anaconda](https://docs.anaconda.com/anaconda/install/windows/)

-Run Anaconda Prompt

-Ensure the conda-forge channel has been added by using "conda config 
--add channels conda-forge
-Create a new conda environment e.g. conda create -n sgFCS_PTU

-Activate the new environment e.g. activate sgFCS_PTU

-Install the following packages using "conda install jupyter numpy scipy numba lmfit emcee matplotlib git"
-Run "pip install git+https://github.com/t-j-murdoch/FCS_point_correlator/"
-Run "Jupyter lab"

For full functionality pip install corner should also be run

## How to cite

Murdoch, T.J., Quienne, B., Pinaud, J., Caillol, S. and Martín-Fabiani, I., 2024. Understanding associative polymer self-assembly with shrinking gate fluorescence correlation spectroscopy. Nanoscale, 16(26), pp.12660-12669.
https://zenodo.org/records/11208305 
