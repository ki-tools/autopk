
# autopk

This library is designed to help automate some of the manual labor involved in 
fitting models to pharmacokinetic data. If you give it a properly formatted csv
and tell it which models to look at it will:

1. Fit each model to each subject's data.
2. Make plots of all the fits, grouped by the experiment variable specified in the data.
3. Make a spreadsheet with fit parameters and BIC values.
4. Tell you which model types are the best fits for (a) each experiment and (b) overall.
5. If covariates are available, will fit a OLS model where the inputs are covariates and outputs are model parameters to help you estimate which covariates affect the fits.


## Installation Instructions

1. Download and install the late version of Anaconda or miniconda.
2. Make sure git is installed.
3. Open the Anaconda command prompt or a terminal where conda is accessible.
4. Make a new virtual environment with `conda create -n autopk python=3.7`
5. Activate the virtual environment with `conda activate autopk`
6. Clone the repo: `git clone https://github.com/ki-tools/autopk.git`
7. Install the repo: `pip install -e autopk/`
8. To confirm that the package installed, run `autopk model-selection --help` and confirm that the output is sensible.


## Data

See `autopk/sample_data/sample_data.csv` for an example csv that shows the format expected by the model selection tool.
Each csv must have the following columns:

- Experiment - This variable represents a grouping of subjects.
- Label - This variable specifies individual subjects. 
- Time - Continuous variable of when the measurements were taken.
- Value - Continuous variable of the measurement.

The rest of the columns will be treated as covariates. These can be something like
sex (binary), weight (continuous), peptide (categorical), and so on. You can include as
many as you like.


## Example

Here's how to run the analysis in the sample_data directory:

`autopk model-selection autopk/sample_data/sample_data.csv --model_names bolus,infusion --compartments 1,2,3 --n_cores 8 --analysis_name autopk/sample_data --overwrite --zeros_as_missing`

Here we want to try bolus and infusion models for 1, 2, and 3 compartments. The computations will run in parallel on 8 cares, 
and the results will be put into the already-existing `autopk/sample_data` directory. This will run for ~5m and will then produce the following files:

- data.pickle - A backup file for the fits and other data structures. Feel free to ignore.
- fits_A.png, fits_B.png - The sample data had two groups: A and B. We have one file of each of the 6 fits for each group.
- residuals_A.png, residuals_B.png - Similar to fits_*.png but with normalized residuals instead.
- fitting_details.csv - Contains the actual fitted parameters for all 6 models for all individual time-series, as well as the BIC values.
- model_selection_probabilities.csv - Probabilities of best model for each group. Note that some columns are missing here - these are just 0 probabilities for all rows.
- OLS_per_parameters_results_* - One file per model. These are the results of the OLS fit from covariate inputs to fitted parameter outputs. Note that the `alpha`, `beta` and `gamma` parameters are converted into half-lives with `log(2) / alpha`.
