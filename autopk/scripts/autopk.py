import click
import numpy as np
import pandas as pd
import pickle
import pathlib
from autopk.models import *
from autopk.plotting import plot_fit_grid
from collections import Counter, defaultdict
from joblib import Parallel, delayed
import statsmodels.formula.api as smf
import statsmodels.api as sm
from io import StringIO


all_models = [    
    calc_sd_1cmt_linear_bolus,
    calc_sd_1cmt_linear_infusion,
    calc_sd_2cmt_linear_bolus,
    calc_sd_2cmt_linear_infusion,
    calc_sd_3cmt_linear_bolus,
    calc_sd_3cmt_linear_infusion
]


def get_right_models(model_names, compartments):
    # check validity of model_names
    model_names = model_names.split(',')
    for model_name in model_names:
        if model_name not in ['bolus', 'infusion', 'oral_1']:
            raise click.BadParameter('Model names all have to be one of bolus, infusion, oral_1!')
    
    # check validity of compartments
    compartments = compartments.split(',')
    for compartment in compartments:
        if compartment not in ['1', '2', '3']:
            raise click.BadParameter('Compartments have to be one of 1, 2, 3!')
    
    # get the valid models
    subset_of_models = []
    for model in all_models:
        has_compartment = np.any([i + 'cmt' in model.__name__ for i in compartments])
        has_model_name = np.any([i in model.__name__ for i in model_names])
        if has_compartment and has_model_name:
            subset_of_models.append(model)
    return subset_of_models


def clean_df(df, zeros_as_missing=True):
    def process_uncertain(s):
        if type(s) is not str:
            if s == 0 and zeros_as_missing:
                return np.nan
            else:
                return s
        elif '>' in s:
            return np.nan
        elif '<' in s:
            return np.nan 
        else: # not sure what this would be
            return np.nan 

    # find 'uncertain' values and fix them up
    df['Value'] = df['Value'].apply(process_uncertain)
    
    # remove rows where the value is missing
    df = df[~pd.isnull(df['Value'])]
    
    return df


def construct_result_csvs(results_dict, exp2label, models):
    label2exp = {}
    for exp, labels in exp2label.items():
        for label in labels:
            label2exp[label] = exp
    
    columns = ['Experiment', 'Label', 'Model', 'BIC']
    for model in models:
        for var in VAR_NAMES[model.__name__]:
            if var not in columns:
                columns.append(var)
    pfits_df = pd.DataFrame(data=None, columns=columns).apply(pd.to_numeric, downcast='float', errors='ignore')

    bics_dict = {}
    i = 0
    for label, (pfit_dict, bic_dict) in results_dict.items():
        bics_dict[label] = bic_dict
        for model_name, pfit in pfit_dict.items():
            pfits_df.loc[i, 'Experiment'] = label2exp[label]
            pfits_df.loc[i, 'Label'] = label
            pfits_df.loc[i, 'Model'] = model_name
            pfits_df.loc[i, 'BIC'] = np.round(bic_dict[model_name], 4)
            for var_name, p in zip(VAR_NAMES[model_name], pfit): 
                pfits_df.loc[i, var_name] = np.round(p, 4)
            i += 1

    bic_df = pd.DataFrame(bics_dict).T.reset_index()
    bic_df.rename(columns={'index': 'Label'}, inplace=True)
    bic_df.insert(0, 'Experiment', bic_df['Label'].apply(lambda s: label2exp[s]))
    bic_df.sort_values(['Experiment', 'Label'], inplace=True)
    bic_df = bic_df.applymap(lambda x: np.round(x, 4) if type(x) is float else x)
    
    per_exp_best = bic_df.groupby('Experiment').idxmin(axis=1).groupby('Experiment').value_counts(normalize=True)
    overall_best = pd.DataFrame(bic_df.groupby('Experiment').idxmin(axis=1).value_counts(normalize=True))
    overall_best = pd.concat({'--Overall--': overall_best}, names=['Experiment'])
    probs_df = pd.concat((per_exp_best, overall_best))
    probs_df.index.names = ['Experiment', 'Model']
    probs_df.columns = ['Probability']
    probs_df = probs_df.unstack(-1, fill_value=0)
    
    return pfits_df, bic_df, probs_df


def fit(df, models, label):
    df_sub = df[df.Label == label]
    x = df_sub['Time'].values
    y = df_sub['Value'].values
    if len(x) > 1:
        pfits = {}
        bics = {}
        for model in models:
            pfit = cma_wrapper(x, y, model)
            pfits[model.__name__] = pfit
            y_pred = model(x, *pfit)
            bics[model.__name__] = BIC(y, y_pred, len(pfit))
        return label, pfits, bics
    else:
        return label, None, None


@click.group()
def cli():
    pass

@click.command()
@click.argument('csv_data_path', type=click.Path(exists=True))
@click.option('--model_names', default='bolus,infusion', show_default=True, help='A comma separated list of models you want to try. Can be: bolus, infusion, oral_1.')
@click.option('--compartments', default='1,2', show_default=True, help='A comma separated list of comparments you want to try. Can be 1, 2, 3.')
@click.option('--n_cores', default=1, show_default=True, help='Number of CPU cores to use on this machine.')
@click.option('--analysis_name', default='analysis', show_default=True, help='Analysis name and also the path where results and figures will be stored.')
@click.option('--zeros_as_missing', is_flag=True, help='Whether to treat 0 values as missing.')
@click.option('--overwrite', is_flag=True, help='Whether to overwrite results in the path specified by analysis_name.')
def model_selection(csv_data_path, model_names, compartments, n_cores, analysis_name, zeros_as_missing, overwrite):
    """Fit a a range of PK models to each animal experiment in CSV_DATA_PATH in your data and count which ones are best.
    Note that CSV_DATA_PATH must have the following column names: 'Experiment', 'Label', 'Time', 'Value'.
    All other columns will be treated as covariates. Don't include other columns or they will be treated as covariates.
    """
    # check if results directory already exists
    analysis_path = pathlib.Path.cwd() / analysis_name
    if not overwrite:
        if pathlib.Path(analysis_path).exists():
            raise click.UsageError(f'Analysis directory already exists. Either specify a new analysis name, or use --overwrite.')
    
    # create the analysis directory
    analysis_path.mkdir(parents=True, exist_ok=True)
        
    models = get_right_models(model_names, compartments)
    print('Performing model selection with the following model types:')
    for model in models: print(model.__name__)
    
    # loading data
    
    df = pd.read_csv(csv_data_path)
    
    # check to see that the needed columns are there
    for col in ['Experiment', 'Label', 'Time', 'Value']:
        if col not in df.columns:
            raise click.UsageError(f'You have to have {col} as one of the column names in your CSV!')
    
    # clean the data a bit
    df = clean_df(df, zeros_as_missing)
    
    # check to make sure the Label doesn't appear in multiple experiments
    label_sets = df.groupby('Experiment')['Label'].apply(set)
    unique_labels = True
    for i in label_sets:
        for j in label_sets:
            if i != j and len(i.intersection(j)) > 0:
                unique_labels = False
                break
    
    if unique_labels is False:
        print('The individual animal time-series labels in the Label column are not unique. Will concatenate with the Experiment column.')
        df['Label'] = df['Experiment'].astype(str) + ' - ' + df['Label'].astype(str)

    all_labels = pd.unique(df['Label'])
    '''
    with open(analysis_path / 'data.pickle', 'rb') as f:
        df, results_dict, exp2label, pfits_df, probs_df = pickle.load(f)
    '''
    # run the parallel fit job
    results = Parallel(n_jobs=n_cores)(delayed(fit)(df, models, label) for label in all_labels)
    results_dict = {}
    for label, pfit_dict, bic_dict in results:
        if pfit_dict is not None:
            results_dict[label] = (pfit_dict, bic_dict)
    
    exp2label = defaultdict(list)
    for exp, label in zip(df['Experiment'], df['Label']):
        if label not in exp2label[exp]: 
            exp2label[exp].append(label)
    
    # plot all results
    for exp, labels in exp2label.items():
        plot_fit_grid(df, exp, labels, results_dict, save_path=analysis_path / f'fits_{exp}.png', plot_type='fit')
        plot_fit_grid(df, exp, labels, results_dict, save_path=analysis_path / f'residuals_{exp}.png', plot_type='residuals')

    # save the parameters and BICs for each experiment / label
    pfits_df, bic_df, probs_df = construct_result_csvs(results_dict, exp2label, models)
    pfits_df.to_csv(analysis_path / 'fitting_details.csv', index=False)
    probs_df.reset_index().to_csv(analysis_path / 'model_selection_probabilities.csv', index=False)
      
    with open(analysis_path / 'data.pickle', 'wb') as f:
        pickle.dump((df, results_dict, exp2label, pfits_df, probs_df), f)
    
    # writing out the original CSV but with predictions and residuals appended
    for model in models:
        params_per_label = np.vstack(df['Label'].apply(lambda label: results_dict[label][0][model.__name__] 
                                                       if label in results_dict
                                                       else [np.nan] * len(VAR_NAMES[model.__name__])))
        x = df['Time'].values
        y = df['Value'].values
        preds = []
        residuals = []
        for i in range(len(x)):
            pred = model(x[i], *params_per_label[i, :])
            preds.append(pred)
            residuals.append( ((y[i] - pred) / y[i]))
        df['Prediction'] = preds
        df['Residual'] = residuals
        df.to_csv(analysis_path / f'predictions_and_residuals_for_{model.__name__}.csv', sep=',', index=False)
    
    print(f'Model selection done. Check the {analysis_path} for model fit figures, fit overview CSV, prediction/residual CSVs, and model selection CSV.')
    
    print(f'Now performing second-stage OLS-based covariate modeling.')
    df_covs = df.drop(['Time', 'Value', 'Prediction', 'Residual'], axis=1).drop_duplicates()
    covariates = list(df_covs.columns)
    covariates.remove('Experiment')
    covariates.remove('Label')
    all_covs = " + ".join(covariates)
    df_covs[covariates] = df_covs[covariates].apply(pd.to_numeric, downcast='float', errors='ignore')

    # normalize the numerical covariates
    numerical_cols = df_covs[covariates].select_dtypes(include=[np.number]).columns.values
    print('Normalizing the following numerical covariate columns:', numerical_cols)
    for col in numerical_cols:
        strictly_positive = np.all(df_covs.loc[:, col].values > 0)
        if strictly_positive:
            df_covs.loc[:, col] = np.log(df_covs.loc[:, col].values)
        df_covs.loc[:, col] -= np.median(df_covs.loc[:, col].values)

    if len(all_covs) == 0:
        print('There are no covariates, so no covariate analysis will be performed.')
    else:
        for model in models:
            pfits_df_sub = pfits_df[pfits_df['Model'] == model.__name__].dropna(axis=1, how='all')
            pfits_df_sub = df_covs.merge(pfits_df_sub, how='inner', on=['Experiment', 'Label'])
            dfs = []
            with open(analysis_path / f'OLS_per_parameters_results_for_{model.__name__}.txt', 'w') as f:
                for y in VAR_NAMES[model.__name__]:
                    y_vec = pfits_df_sub[y]
                    y_vec = y_vec.apply(pd.to_numeric, downcast='float', errors='ignore').values
                    if y in ['alpha', 'beta', 'gamma']:
                        output = 'log_thalf_' + y
                        pfits_df_sub[output] = np.log1p(np.log(2) / y_vec)
                    else:
                        output = 'log_' + y
                        pfits_df_sub[output] = np.log1p(y_vec)
                    pfits_df_sub[output] = pfits_df_sub[output].astype(float)
                    res = smf.ols(formula=f'{output} ~ {all_covs}', data=pfits_df_sub).fit()
                    f.write(str(res.summary(title=f'OlS Results: {output} ~ {all_covs}')) + '\n' * 5)
                    results_df = pd.read_csv(StringIO(res.summary().tables[1].as_csv()), index_col=0)
                    results_df.index = [y + ' ' + i for i in results_df.index]
                    dfs.append(results_df)
            results_single_df = pd.concat(dfs)
            results_single_df.to_csv(analysis_path / f'OLS_per_parameters_results_for_{model.__name__}.csv', sep=',')

        print(f'OLS covariate analysis done. Check the {analysis_path} for coefficient summaries.')


cli.add_command(model_selection)


if __name__ == '__main__':
    cli()