

Installation instructions:

(1) Download and install Anaconda/miniconda.
(2) Make sure git is installed.
(3) Open the Anaconda command prompt or a terminal where conda is accessible.
(4) Make a new virtual environment with `conda create -n autopk python=3.7`
(5) Activate the virtual environment with `conda activate autopk`
(6) Clone the repo: `git clone https://github.com/ki-tools/autopk.git`
(7) Navigate to the repo folder: `cd autopk`
(8) Install the repo: `pip install -e .`
(9) To confirm that the package installed, run `autopk model-selection --help` and confirm that the output is sensible.