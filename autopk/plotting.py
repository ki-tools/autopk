import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk')


def factor_int(n):
    val = np.ceil(np.sqrt(n))
    val2 = np.ceil(n / val)
    if val > val2:
        return int(val2), int(val)
    else:
        return int(val), int(val2)
    

def plot_fit_grid(df, exp, labels, results_dict, save_path, plot_type='fit'):
    n_rows, n_cols = factor_int(len(labels))
    plt.figure(figsize=(n_rows * 6, n_cols * 6), tight_layout=True)
    for i, label in enumerate(labels):
        pfit_dict, bic_dict = results_dict[label]
        df_sub = df[df.Label == label]
        x = df_sub['Time'].values
        y = df_sub['Value'].values
        x_range = np.linspace(x.min(), x.max(), 1000)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(f'Label={label}')
        if plot_type == 'fit':
            if i == 0:
                plt.scatter(x, y, label='Data')
            else:
                plt.scatter(x, y)
        for model_name, model_params in pfit_dict.items():
            bic = bic_dict[model_name]
            if i == 0:
                label = model_name
            else:
                label = None
            if plot_type == 'fit':
                y_fit = globals()[model_name](x_range, *model_params)
                plt.semilogy(x_range, y_fit, alpha=0.7, label=label)
            else:
                y_fit = globals()[model_name](x, *model_params)
                plt.plot(x, (y - y_fit) / y, alpha=0.7, label=label)

        if plot_type == 'fit':
            plt.ylim([y.min() * 0.1, y.max() * 10])
            plt.ylabel('Value')
        else:
            plt.ylabel('(Data - Fit) / Data')
        plt.xlabel('Time')
        
    plt.figlegend(loc='center left', bbox_to_anchor=(1, 0.5))
    if plot_type == 'fit':
        plt.suptitle(f'Fits for {exp}', y=1.03)
    else:
        plt.suptitle(f'Relative Residuals for {exp}', y=1.03)
        
    plt.savefig(save_path, bbox_inches="tight")