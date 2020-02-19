

Installation instructions:

(1) If you don't have Python 3.7+, then download and install Anaconda/miniconda.

(2) Make sure git is installed.

(3) Open the Anaconda command prompt or a terminal where conda is accessible.

(4) Make a new virtual environment with `conda create -n autopk python=3.7`

(5) Activate the virtual environment with `conda activate autopk`

(6) Clone the repo to a directory of your choice: `git clone https://github.com/ki-tools/autopk.git`

(7) Navigate to the repo directory: `cd autopk`

(8) Install the repo: `pip install -e .`

(9) To confirm that the package installed, run `autopk model-selection --help` and confirm that the output is sensible.
