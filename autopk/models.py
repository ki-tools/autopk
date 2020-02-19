# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:22:13 2020

@author: Sergey

Equation numbers refer to PKPDlibrary.pdf
"""

import numpy as np
import cma


# to do: consider constraints
# by 10 * duration of experiment the concentration should be 0 for every curve
# ka > alpha > beta > gamma


VAR_NAMES = {    
    'calc_sd_1cmt_linear_bolus': ['A', 'alpha'],
    'calc_sd_1cmt_linear_infusion': ['A', 'alpha', 'Tinf'],
    'calc_sd_1cmt_linear_oral_1': ['A', 'alpha', 'ka'],
    'calc_sd_1cmt_linear_oral_1_lag': ['A', 'alpha', 'ka', 'Tlag'],
    'calc_sd_1cmt_linear_oral_0': ['A', 'alpha', 'Tk0'],
    'calc_sd_1cmt_linear_oral_0_lag': ['A', 'alpha', 'Tlag', 'Tk0'],
    'calc_sd_2cmt_linear_bolus': ['A', 'alpha', 'B', 'beta'],
    'calc_sd_2cmt_linear_infusion': ['A', 'alpha', 'B', 'beta', 'Tinf'],
    'calc_sd_2cmt_linear_oral_1': ['A', 'alpha', 'B', 'beta', 'ka'],
    'calc_sd_2cmt_linear_oral_1_lag': ['A', 'alpha', 'B', 'beta', 'ka', 'Tlag'],
    'calc_sd_2cmt_linear_oral_0': ['A', 'alpha', 'B', 'beta', 'Tk0'],
    'calc_sd_2cmt_linear_oral_0_lag': ['A', 'alpha', 'B', 'beta', 'Tlag', 'Tk0'],
    'calc_sd_3cmt_linear_bolus': ['A', 'alpha', 'B', 'beta', 'C', 'gamma'],
    'calc_sd_3cmt_linear_infusion': ['A', 'alpha', 'B', 'beta', 'C', 'gamma', 'Tinf'],
    'calc_sd_3cmt_linear_oral_1': ['A', 'alpha', 'B', 'beta', 'C', 'gamma', 'ka'],
    'calc_sd_3cmt_linear_oral_1_lag': ['A', 'alpha', 'B', 'beta', 'C', 'gamma', 'ka', 'Tlag'],
    'calc_sd_3cmt_linear_oral_0': ['A', 'alpha', 'B', 'beta', 'C', 'gamma', 'Tk0'],
    'calc_sd_3cmt_linear_oral_0_lag': ['A', 'alpha', 'B', 'beta', 'C', 'gamma', 'Tlag', 'Tk0']
}


def weighted_mean_squares(y_true, y_pred, log=True):
    y_pred = np.maximum(y_pred, 1e-12)
    if log:
        y_true = np.log1p(y_true)
        y_pred = np.log1p(y_pred)
    weights = 1 / (y_true + 1e-6)
    squares = (y_true - y_pred) ** 2
    return np.average(squares, weights=weights)


def BIC(y_true, y_pred, k, log=True):
    mss = weighted_mean_squares(y_true, y_pred, log)
    n = len(y_pred)
    return k * np.log(n) + n * np.log(mss)


def AIC(y_true, y_pred, k, log=True):
    mss = weighted_mean_squares(y_true, y_pred, log)
    n = len(y_pred)
    return 2 * k + n * np.log(mss)


def estimate_bounds_and_initial_params(x, y, model_name):
    '''
    1.   Find the largest positive concentration in the data set and its corresponding time.
    2.   The lag time cannot be larger than say 2x this time.
    3.   The intercept parameters (A,B,C) cannot be larger than say 2x this maximal concentration value.
    4.   Following the data forward until the concentration is ½ of the maximal (ie. observed half-life) and noting the elapsed time.
    5.   The decay parameters (α, ß, Γ) cannot be larger than say 2x log(2)/(elapsed time). 
    6.   Following the data back until the concentration is ½ of the maximal (ie. observed half-life) and noting the elapsed time.
    7.   The absorption parameter (ka) cannot be larger than say 2x log(2)/(elapsed time).
    8.   The infusion time Tk0 or Tinf should not be more than 2x the time it takes to reach maximal concentration.
    '''
    upper_bounds_dict = {}
    lower_bounds_dict = {}
    argmax_y = np.argmax(y)
    upper_bounds_dict['Tk0'] = 2 * x[argmax_y]
    upper_bounds_dict['Tinf'] = x[argmax_y + 1]
    upper_bounds_dict['Tlag'] = x[argmax_y + 1]
    upper_bounds_dict['A'] = 2 * y[argmax_y]
    upper_bounds_dict['B'] = 2 * y[argmax_y]
    upper_bounds_dict['C'] = 2 * y[argmax_y]
    lower_bounds_dict['Tk0'] = x[argmax_y] / 10
    lower_bounds_dict['Tinf'] = x[argmax_y] / 10
    lower_bounds_dict['Tlag'] = x[argmax_y] / 10
    lower_bounds_dict['A'] = 0
    lower_bounds_dict['B'] = 0
    lower_bounds_dict['C'] = 0
    
    forward_flag = x > x[argmax_y]
    x_forward = x[forward_flag]
    y_forward = y[forward_flag]
    half_loc_forward = np.argmin(np.abs(y_forward - 0.5 * y[argmax_y]))
    x_halflife_forward = x_forward[half_loc_forward]
    upper_bounds_dict['alpha'] = 4 * np.log(2) / (x_halflife_forward - x[argmax_y])
    upper_bounds_dict['beta'] = upper_bounds_dict['alpha']
    upper_bounds_dict['gamma'] = upper_bounds_dict['alpha']
    lower_bounds_dict['alpha'] = np.log(2) / (10 * np.max(x))
    lower_bounds_dict['beta'] = lower_bounds_dict['alpha']
    lower_bounds_dict['gamma'] = lower_bounds_dict['alpha']

    backward_flag = x < x[argmax_y]
    x_backward = x[backward_flag]
    if len(x_backward) > 0:
        y_backward = y[backward_flag]
        half_loc_backward = np.argmin(np.abs(y_backward - 0.5 * y[argmax_y]))
        upper_bounds_dict['ka'] = 8 * np.log(2) / (x[argmax_y] - x_backward[half_loc_backward])
    else:
        upper_bounds_dict['ka'] = 2 * upper_bounds_dict['alpha']
    lower_bounds_dict['ka'] = 0
    
    upper_bounds = np.array([upper_bounds_dict[i] for i in VAR_NAMES[model_name]])
    lower_bounds = np.array([lower_bounds_dict[i] for i in VAR_NAMES[model_name]])
    p_init = (upper_bounds - lower_bounds) / 2
    return upper_bounds, lower_bounds, p_init


def cma_wrapper(x, y, model, std=0.1, popsize=5, restarts=5, log=True):
    loss_f = lambda p: weighted_mean_squares(y, model(x, *p), log)
    upper_bounds, lower_bounds, p_init = estimate_bounds_and_initial_params(x, y, model.__name__)
    result = cma.fmin(
        loss_f,
        f'np.random.uniform({list(lower_bounds)}, {list(upper_bounds)})', 
        std,
        {
            'bounds': [lower_bounds, upper_bounds],
            'popsize': popsize,
            'verbose': -9,
            'CMA_stds': upper_bounds / 4
        },
        restarts=restarts,
        bipop=True
    )
    
    return order_pfit(result[0], VAR_NAMES[model.__name__])


def order_pfit(pfit, var_names):
    if 'beta' in var_names:
        A_ind = var_names.index('A')
        alpha_ind = var_names.index('alpha')
        B_ind = var_names.index('B')
        beta_ind = var_names.index('beta')
        if pfit[beta_ind] > pfit[alpha_ind]:
            pfit[A_ind], pfit[alpha_ind], pfit[B_ind], pfit[beta_ind] = pfit[B_ind], pfit[beta_ind], pfit[A_ind], pfit[alpha_ind]
    if 'gamma' in var_names:
        C_ind = var_names.index('C')
        gamma_ind = var_names.index('gamma')
        if pfit[gamma_ind] > pfit[alpha_ind]:
            pfit[A_ind], pfit[alpha_ind], pfit[C_ind], pfit[gamma_ind] = pfit[C_ind], pfit[gamma_ind], pfit[A_ind], pfit[alpha_ind]
        if pfit[gamma_ind] > pfit[beta_ind]:
            pfit[B_ind], pfit[beta_ind], pfit[C_ind], pfit[gamma_ind] = pfit[C_ind], pfit[gamma_ind], pfit[B_ind], pfit[beta_ind]
    return pfit


def calc_sd_1cmt_linear_bolus(x, *p):
    # 1.1
    A, alpha = p
    return A * np.exp(-alpha * x)
    
    
def calc_sd_1cmt_linear_infusion(x, *p):
    # 1.6
    A, alpha, Tinf = p
    scale = A / (Tinf * alpha)
    result = np.zeros(x.shape)
    flag = x <= Tinf
    result[flag] = scale * (1 - np.exp(-alpha * x[flag]))
    result[~flag] = scale * (1 - np.exp(-alpha * Tinf)) * np.exp(-alpha * (x[~flag] - Tinf))
    return result
    
    
def calc_sd_1cmt_linear_oral_1(x, *p):
    # 1.11
    A, alpha, ka = p
    scale = A * ka / (ka - alpha)
    term1 = np.exp(-alpha * x)
    term2 = np.exp(-ka * x)
    return scale * (term1 - term2)
    

def calc_sd_1cmt_linear_oral_1_lag(x, *p):
    # 1.14
    A, alpha, ka, Tlag = p
    flag = x <= Tlag
    result = np.zeros(x.shape)
    result[flag] = 0  # mostly for documentation
    scale = A * ka / (ka - alpha)
    term1 = np.exp(-alpha * (x[~flag] - Tlag))
    term2 = np.exp(-ka * (x[~flag] - Tlag))
    result[~flag] = scale * (term1 - term2)
    return result
    
    
def calc_sd_1cmt_linear_oral_0(x, *p):
    # 1.21
    A, alpha, Tk0 = p
    scale = A / (Tk0 * alpha)
    flag = x <= Tk0
    result = np.zeros(x.shape)
    result[flag] = scale * (1 - np.exp(-alpha * x[flag]))
    result[~flag] = scale * (1 - np.exp(-alpha * Tk0)) * np.exp(-alpha * (x[~flag] - Tk0))
    return result


def calc_sd_1cmt_linear_oral_0_lag(x, *p):
    # 1.24
    A, alpha, Tlag, Tk0 = p
    result = np.zeros(x.shape)
    scale = A / (Tk0 * alpha)
    flag = (Tlag < x ) & (x <= Tlag + Tk0)
    result[flag] = scale * (1 - np.exp(-alpha * (x[flag] - Tlag)))
    result[~flag] = scale * (1 - np.exp(-alpha * Tk0)) * np.exp(-alpha * (x[~flag] - Tlag - Tk0))
    result[x <= Tlag] = 0
    return result
    
    
def calc_sd_2cmt_linear_bolus(x, *p):
    # 1.31
    A, alpha, B, beta = p
    return A * np.exp(-alpha * x) + B * np.exp(-beta * x)
    

def calc_sd_2cmt_linear_infusion(x, *p):
    # 1.36
    A, alpha, B, beta, Tinf = p
    scale1 = (A / (Tinf * alpha))
    scale2 = (B / (Tinf * beta))
    result = np.zeros(x.shape)
    flag = x <= Tinf
    term1 = (1 - np.exp(-alpha * x[flag]))
    term2 = (1 - np.exp(-beta * x[flag]))
    result[flag] = scale1 * term1 + scale2 * term2
    term1 = (1 - np.exp(-alpha * Tinf)) * np.exp(-alpha * (x[~flag] - Tinf))
    term2 = (1 - np.exp(-beta * Tinf)) * np.exp(-beta * (x[~flag] -  Tinf))
    result[~flag] = scale1 * term1 + scale2 * term2
    return result
    
    
def calc_sd_2cmt_linear_oral_1(x, *p):
    # 1.41
    A, alpha, B, beta, ka = p
    return A * np.exp(-alpha * x) + B * np.exp(-beta * x) \
        - (A + B) * np.exp(-ka * x)
    

def calc_sd_2cmt_linear_oral_1_lag(x, *p):
    # 1.44
    A, alpha, B, beta, ka, Tlag = p
    result = A * np.exp(-alpha * (x - Tlag)) + B * np.exp(-beta * (x - Tlag)) \
        - (A + B) * np.exp(-ka * (x - Tlag))
    result[x <= Tlag] = 0
    return result


def calc_sd_2cmt_linear_oral_0(x, *p):
    # 1.51
    A, alpha, B, beta, Tk0 = p
    scale1 = (A / (Tk0 * alpha))
    scale2 = (B / (Tk0 * beta))
    result = np.zeros(x.shape)
    flag = x <= Tk0
    term1 = (1 - np.exp(-alpha * x[flag]))
    term2 = (1 - np.exp(-beta * x[flag]))
    result[flag] = scale1 * term1 + scale2 * term2
    term1 = (1 - np.exp(-alpha * Tk0)) * np.exp(-alpha * (x[~flag] - Tk0))
    term2 = (1 - np.exp(-beta * Tk0)) * np.exp(-beta * (x[~flag] - Tk0))
    result[~flag] = scale1 * term1 + scale2 * term2
    return result


def calc_sd_2cmt_linear_oral_0_lag(x, *p):
    # 1.54
    A, alpha, B, beta, Tlag, Tk0 = p
    result = np.zeros(x.shape)
    scale1 = A / (Tk0 * alpha)
    scale2 = B / (Tk0 * beta)
    flag = (Tlag < x ) & (x <= Tlag + Tk0)
    term1 = (1 - np.exp(-alpha * (x[flag] - Tlag)))
    term2 = (1 - np.exp(-beta * (x[flag] - Tlag)))
    result[flag] = scale1 * term1 + scale2 * term2
    term1 = (1 - np.exp(-alpha * Tk0)) * np.exp(-alpha * (x[~flag] - Tlag - Tk0))
    term2 = (1 - np.exp(-beta * Tk0)) * np.exp(-beta * (x[~flag] - Tlag - Tk0)) 
    result[~flag] = scale1 * term1 + scale2 * term2
    result[x <= Tlag] = 0
    return result
    
    
def calc_sd_3cmt_linear_bolus(x, *p):
    # 1.61
    A, alpha, B, beta, C, gamma = p
    return A * np.exp(-alpha * x) + B * np.exp(-beta * x) \
        + C * np.exp(-gamma * x)
    

def calc_sd_3cmt_linear_infusion(x, *p):
    # 1.66
    A, alpha, B, beta, C, gamma, Tinf = p
    result = np.zeros(x.shape)
    scale1 = (A / (Tinf * alpha))
    scale2 = (B / (Tinf * beta))
    scale3 = (C / (Tinf * gamma))
    flag = x <= Tinf
    term1 = (1 - np.exp(-alpha * x[flag]))
    term2 = (1 - np.exp(-beta * x[flag]))
    term3 = (1 - np.exp(-gamma * x[flag]))
    result[flag] = scale1 * term1 + scale2 * term2 + scale3 * term3
    term1 = (1 - np.exp(-alpha * Tinf)) * np.exp(-alpha * (x[~flag] - Tinf))
    term2 = (1 - np.exp(-beta * Tinf)) * np.exp(-beta * (x[~flag] - Tinf))
    term3 = (1 - np.exp(-gamma * Tinf)) * np.exp(-gamma * (x[~flag] - Tinf))
    result[~flag] = scale1 * term1 + scale2 * term2 + scale3 * term3
    return result
    

def calc_sd_3cmt_linear_oral_1(x, *p):
    # 1.71
    A, alpha, B, beta, C, gamma, ka = p
    return A * np.exp(-alpha * x) + B * np.exp(-beta * x) \
        + C * np.exp(-gamma * x) - (A + B + C) * np.exp(-ka * x)


def calc_sd_3cmt_linear_oral_1_lag(x, *p):
    # 1.74
    A, alpha, B, beta, C, gamma, ka, Tlag = p
    result = A * np.exp(-alpha * (x - Tlag)) + B * np.exp(-beta * (x - Tlag)) \
        + C * np.exp(-gamma * (x - Tlag)) - (A + B + C) * np.exp(-ka * (x - Tlag))
    result[x <= Tlag] = 0
    return result


def calc_sd_3cmt_linear_oral_0(x, *p):
    # 1.84
    A, alpha, B, beta, C, gamma, Tk0 = p
    result = np.zeros(x.shape)
    scale1 = A / (Tk0 * alpha)
    scale2 = B / (Tk0 * beta)
    scale3 = C / (Tk0 * gamma)
    flag = (x <= Tk0)
    term1 = (1 - np.exp(-alpha * x[flag]))
    term2 = (1 - np.exp(-beta * x[flag]))
    term3 = (1 - np.exp(-gamma * x[flag]))
    result[flag] = scale1 * term1 + scale2 * term2 + scale3 * term3
    term1 = (1 - np.exp(-alpha * Tk0)) * np.exp(-alpha * (x[~flag] - Tk0))
    term2 = (1 - np.exp(-beta * Tk0)) * np.exp(-beta * (x[~flag] - Tk0)) 
    term3 = (1 - np.exp(-gamma * Tk0)) * np.exp(-gamma * (x[~flag] - Tk0)) 
    result[~flag] = scale1 * term1 + scale2 * term2 + scale3 * term3
    return result

    
def calc_sd_3cmt_linear_oral_0_lag(x, *p):
    # 1.84
    A, alpha, B, beta, C, gamma, Tlag, Tk0 = p
    result = np.zeros(x.shape)
    scale1 = A / (Tk0 * alpha)
    scale2 = B / (Tk0 * beta)
    scale3 = C / (Tk0 * gamma)
    flag = (Tlag < x) & (x <= Tlag + Tk0)
    term1 = (1 - np.exp(-alpha * (x[flag] - Tlag)))
    term2 = (1 - np.exp(-beta * (x[flag] - Tlag)))
    term3 = (1 - np.exp(-gamma * (x[flag] - Tlag)))
    result[flag] = scale1 * term1 + scale2 * term2 + scale3 * term3
    term1 = (1 - np.exp(-alpha * Tk0)) * np.exp(-alpha * (x[~flag] - Tlag - Tk0))
    term2 = (1 - np.exp(-beta * Tk0)) * np.exp(-beta * (x[~flag] - Tlag - Tk0)) 
    term3 = (1 - np.exp(-gamma * Tk0)) * np.exp(-gamma * (x[~flag] - Tlag - Tk0)) 
    result[~flag] = scale1 * term1 + scale2 * term2 + scale3 * term3
    result[x <= Tlag] = 0
    return result