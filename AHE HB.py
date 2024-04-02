import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import r2_score
import os
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
import itertools
from ordered_set import OrderedSet

# Create a function to format y-axis labels


def format_func(value, tick_number):
    return f'{value:.12f}'


# %% PARAMETERS TO GO BETWEEN HALL BAR AND CORBINO SECTOR and some constants:
fixer_HB = 1.14325  # FOR scaling the coefficient, so that Rl(coef=1) = 1Ohm

fixer = fixer_HB



###path_HB = Specify the path to the file that is being read

hall_resistance_HB = 121.21212 #Value of the resistance of the Hall's Bar from simulation


current_device = 0.00053


# %% Making of df_HB:

df_HB = pd.DataFrame(columns=['Rl', 'Theta', 'pow_0', 'pow_load', 'avg_cur',
                     'volt_dif', 'charge_acc', 'pow_ratio', 'alpha', 'alpha_scaled', 'Rl_scaled'])
# charge_acc is actually the difference in charge accumulation on the two sides
list_of_thetas_for_columns_HB = []
list_of_Rl_for_columns_HB = []

# READING OF data HB
with open(path_HB, 'r') as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if the line starts with '% Data (P0 (W) @'
        if line.startswith('% Data (pow_0 (W) @'):
            tita = line.split('Theta=')[1].split(',')[0].replace(")", "")
            theta = float(tita)
            cof = line.split('coef=')[1].split(',')[0].replace(")", "")
            Rl = float(cof)*fixer
            list_of_thetas_for_columns_HB.append(theta)
            list_of_Rl_for_columns_HB.append(Rl)
        i = i+1

list_of_thetas_for_columns_HB_unique = list(OrderedSet(
    list_of_thetas_for_columns_HB))  # LOOKING FOR UNIQUE THETAS HB


list_of_Rl_for_columns_HB_unique = list(OrderedSet(
    list_of_Rl_for_columns_HB))  # LOOKING FOR UNIQUE Rl's HB

all_combinations_HB = list(itertools.product(
    list_of_Rl_for_columns_HB_unique, list_of_thetas_for_columns_HB_unique))

df_temp_HB = pd.DataFrame(all_combinations_HB, columns=['Rl', 'Theta'])
df_HB[['Rl', 'Theta']] = df_temp_HB

# POPULATING df_HB
with open(path_HB, 'r') as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if the line starts with '% Data (P0 (W) @'
        if line.startswith('% Data (pow_0 (W) @'):
            tita = line.split('Theta=')[1].split(',')[0].replace(")", "")
            theta = float(tita)
            cof = line.split('coef=')[1].split(',')[0].replace(")", "")
            Rl = float(cof)*fixer
            # convert the string to float
            pow_0_value = float(lines[i+1].strip())
            df_HB.loc[(df_HB['Theta'] == theta) & (
                df_HB['Rl'] == Rl), 'pow_0'] = pow_0_value

        elif line.startswith('% Data (0.5*(cur_down + cur_up)'):
            tita = line.split('Theta=')[1].split(',')[0].replace(")", "")
            theta = float(tita)
            cof = line.split('coef=')[1].split(',')[0].replace(")", "")
            Rl = float(cof)*fixer
            # convert the string to float
            avg_cur_value = float(lines[i+1].strip())
            df_HB.loc[(df_HB['Theta'] == theta) & (
                df_HB['Rl'] == Rl), 'avg_cur'] = avg_cur_value

        elif line.startswith('% Data (volt_up - volt_down (V) @'):
            tita = line.split('Theta=')[1].split(',')[0].replace(")", "")
            theta = float(tita)
            cof = line.split('coef=')[1].split(',')[0].replace(")", "")
            Rl = float(cof)*fixer
            volt_dif_value = float(lines[i+1].strip())
            df_HB.loc[(df_HB['Theta'] == theta) & (
                df_HB['Rl'] == Rl), 'volt_dif'] = volt_dif_value

        elif line.startswith('% Data (charge_up - charge_down'):
            tita = line.split('Theta=')[1].split(',')[0].replace(")", "")
            theta = float(tita)
            cof = line.split('coef=')[1].split(',')[0].replace(")", "")
            Rl = float(cof)*fixer
            charge_acc_value = (float(lines[i+1].strip()))
            df_HB.loc[(df_HB['Theta'] == theta) & (
                df_HB['Rl'] == Rl), 'charge_acc'] = charge_acc_value

        elif line.startswith('% Data (pow_load (W)'):
            tita = line.split('Theta=')[1].split(',')[0].replace(")", "")
            theta = float(tita)
            cof = line.split('coef=')[1].split(',')[0].replace(")", "")
            Rl = float(cof)*fixer
            power_load_value = float(lines[i+1].strip())
            df_HB.loc[(df_HB['Theta'] == theta) & (
                df_HB['Rl'] == Rl), 'pow_load'] = power_load_value

        i = i+1

# MAKING THE SCALING FOR HB
df_HB['pow_load'] = df_HB['pow_load'].astype(float)

pow_0_HB_for_ratio = df_HB[df_HB['Theta'] ==
                           0].loc[df_HB[df_HB['Theta'] == 0]['pow_0'].idxmax(), 'pow_0']

df_HB['pow_ratio'] = df_HB['pow_load'] / float(pow_0_HB_for_ratio)

df_HB['alpha'] = (hall_resistance_HB / df_HB['Rl'])


scaling_factor_HB = df_HB[df_HB['Theta'] ==
                          0.12].loc[df_HB[df_HB['Theta'] == 0.12]['pow_load'].idxmax(), 'alpha']

df_HB['alpha_scaled'] = (df_HB['alpha'] / scaling_factor_HB)

df_HB['Rl_scaled'] = (df_HB['Rl'] / scaling_factor_HB)



# %% Defining desired lengths:
# THETAS

Thetas_for_few_fits = [0.04, 0.08, 0.12, 0.16, 0.2]

Alphas_for_few_fits = [ df_HB['alpha'][372], df_HB['alpha'][420], df_HB['alpha'][720], df_HB['alpha'][997], df_HB['alpha'][1359] ]

Alphas_scaled_for_few_fits = [ df_HB['alpha_scaled'][6], df_HB['alpha_scaled'][420], df_HB['alpha_scaled'][720], df_HB['alpha_scaled'][997], df_HB['alpha_scaled'][1359] ]


# %% DEFINING FITTING FUNCTIONS:

def func_pow_ratio_alpha_fit(x, a, c):  # SAME
    return (a*x)/(1+x)**2 + c


def func_pow_ratio_theta_fit(x, a, c):  # DIFFERENT
    return a*x**2/(1+x**2) + c


def func_cur_y_theta_fit(x, a, c):  # SAME
    return a*x + c


def func_cur_y_alpha_fit(x, a, c):  # SAME
    return a*x/(1+x) + c
    # return a*x + c


def func_charge_alpha_fit(x, a, c):  # SAME
    return a*x/(1+x) + c


def func_charge_theta_fit(x, a, c):  # DIFFERENT
    return a*x/(1+x**2) + c


def func_volt_dif_theta_fit(x, a, c):
    return a*x + c


def func_volt_dif_alpha_fit(x, a, c):
    return a/(1+x) + c

# Fits with no offset


def func_pow_ratio_alpha_fit_no(x, a):  # SAME
    return (a*x)/(1+x)**2


def func_pow_ratio_theta_fit_no(x, a):  # DIFFERENT
    return a*x**2/(1+x**2)


def func_cur_y_theta_fit_no(x, a):  # SAME
    return a*x


def func_cur_y_alpha_fit_no(x, a):  # SAME
    return a*x/(1+x)
    # return a*x


def func_charge_alpha_fit_no(x, a):  # SAME
    return a*x/(1+x)


def func_charge_theta_fit_no(x, a):  # DIFFERENT
    return a*x/(1+x**2)


def func_volt_dif_alpha_fit_no(x, a):
    return a/(1+x)


def func_volt_dif_theta_fit_no(x, a):
    return a*x


def Average(lst):
    return sum(lst) / len(lst)




# %% Data fitting func


def fit_data(subset_x, subset_y, func, initial_guess=[1, 1]):

    popt_pos, pcov_pos = curve_fit(func, subset_x, subset_y, p0=initial_guess)
    y_pred_pos = func(subset_x, *popt_pos)
    r2_pos = r2_score(subset_y, y_pred_pos)

    # Fit the function to the data with negative initial guess
    popt_neg, pcov_neg = curve_fit(
        func, subset_x, subset_y, p0=[-x for x in initial_guess])
    y_pred_neg = func(subset_x, *popt_neg)
    r2_neg = r2_score(subset_y, y_pred_neg)

    # Choose the fit with the higher R^2 score
    if r2_pos > r2_neg:
        popt, pcov, y_pred, r2 = popt_pos, pcov_pos, y_pred_pos, r2_pos
    else:
        popt, pcov, y_pred, r2 = popt_neg, pcov_neg, y_pred_neg, r2_neg

    return popt, pcov, y_pred, r2


# %% Data fitting func_no

def fit_data_no(subset_x, subset_y, func, func_no, initial_guess=[1, 1]):

    popt_pos, pcov_pos = curve_fit(func, subset_x, subset_y, p0=initial_guess)
    y_pred_pos = func(subset_x, *popt_pos)
    r2_pos = r2_score(subset_y, y_pred_pos)

    # Fit the function to the data with negative initial guess
    popt_neg, pcov_neg = curve_fit(
        func, subset_x, subset_y, p0=[-x for x in initial_guess])
    y_pred_neg = func(subset_x, *popt_neg)
    r2_neg = r2_score(subset_y, y_pred_neg)

    # Choose the fit with the higher R^2 score
    if r2_pos > r2_neg:
        popt, pcov, y_pred, r2 = popt_pos, pcov_pos, y_pred_pos, r2_pos
    else:
        popt, pcov, y_pred, r2 = popt_neg, pcov_neg, y_pred_neg, r2_neg
    a, c = popt

    subset_y_no = subset_y - c

    popt_pos_no, pcov_pos_no = curve_fit(
        func_no, subset_x, subset_y_no, p0=initial_guess[0])
    y_pred_pos_no = func_no(subset_x, *popt_pos_no)
    r2_pos_no = r2_score(subset_y_no, y_pred_pos_no)

    popt_neg_no, pcov_neg_no = curve_fit(
        func_no, subset_x, subset_y_no, p0=-initial_guess[0])
    y_pred_neg_no = func_no(subset_x, *popt_neg_no)
    r2_neg_no = r2_score(subset_y_no, y_pred_neg_no)

    if r2_pos_no > r2_neg_no:
        popt_no, pcov_no, y_pred_no, r2_no = popt_pos_no, pcov_pos_no, y_pred_pos_no, r2_pos_no
    else:
        popt_no, pcov_no, y_pred_no, r2_no = popt_neg_no, pcov_neg_no, y_pred_neg_no, r2_neg_no

    return popt_no, pcov_no, y_pred_no, r2_no


# %% GRAPHING HB Power functions:

### Graphing power vs alpha with offset
colors_alpha_fit = cm.rainbow(np.linspace(0, 1, len(Thetas_for_few_fits)))

r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset= df_HB[df_HB['Theta'] == tita]

    popt, pcov, y_pred, r2 = fit_data(subset['alpha'], subset['pow_ratio'], func_pow_ratio_alpha_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt

    x = np.linspace(min(subset['alpha']), max(subset['alpha']), 100)
    y_pred = func_pow_ratio_alpha_fit(subset['alpha'], *popt)
    plt.scatter(subset['alpha'], subset['pow_ratio'], color=color, label=f'\u03F4 = {tita}')
    # Plot the fitted function with color
    plt.plot(x, func_pow_ratio_alpha_fit(x, a, c), color=color, label= f'r2={r2},\n FIT={a}*x/(1+x)**2 + {c}' )

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1', fontsize=25)
plt.ylabel('Plat/P0', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Plat/P0 vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n With Offset \n FIT= a*x**2/(1+x**2) + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

###NO OFFSET Plat/P0 alpha fit

r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset = df_HB[df_HB['Theta'] == tita]
    
    popt, pcov, y_pred, r2 = fit_data(subset['alpha'], subset['pow_ratio'], func_pow_ratio_alpha_fit, initial_guess=[1, 1])
    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['alpha'], subset['pow_ratio'], func_pow_ratio_alpha_fit, func_pow_ratio_alpha_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'pow_ratio'] = subset['pow_ratio'] - c
    
    x = np.linspace(min(subset['alpha']), max(subset['alpha']), 100)
    y_pred = func_pow_ratio_alpha_fit_no(subset['alpha'], a)
    plt.scatter(subset['alpha'], subset['pow_ratio'], color=color, label=f'\u03F4 = {tita}')
    plt.plot(x, func_pow_ratio_alpha_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT={a}*x/(1+x)**2')

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1', fontsize=25)
plt.ylabel('Plat/P0', fontsize=25)
srednie_r2_alpha_fit = Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Plat/P0 vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n No Offset \n FIT= a*x**2/(1+x**2) ', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()


## Graphing power vs alpha_scaled with offset

r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset= df_HB[df_HB['Theta'] == tita]
    popt, pcov, y_pred, r2 = fit_data(subset['alpha_scaled'], subset['pow_ratio'], func_pow_ratio_alpha_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt
    x = np.linspace(min(subset['alpha_scaled']), max(subset['alpha_scaled']), 100)
    y_pred = func_pow_ratio_alpha_fit(subset['alpha_scaled'], *popt)
    plt.scatter(subset['alpha_scaled'], subset['pow_ratio'], color=color, label=f'\u03F4 = {tita}')
    # Plot the fitted function with color
    plt.plot(x, func_pow_ratio_alpha_fit(x, a, c), color=color, label= f'r2={r2},\n FIT={a}*x/(1+x)**2 + {c}' )
plt.axvline(x=1, color='red', linestyle='--', label='\u03B1')
avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1 scaled', fontsize=25)
plt.ylabel('Plat/P0', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Plat/P0 vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n With Offset \n FIT= a*x**2/(1+x**2) + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

### Graphing power vs alpha_scaled NO offset
r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset = df_HB[df_HB['Theta'] == tita]
    
    popt, pcov, y_pred, r2 = fit_data(subset['alpha_scaled'], subset['pow_ratio'], func_pow_ratio_alpha_fit, initial_guess=[1, 1])
    a, c = popt
    
    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['alpha_scaled'], subset['pow_ratio'], func_pow_ratio_alpha_fit, func_pow_ratio_alpha_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'pow_ratio'] = subset['pow_ratio'] - c
    
    x = np.linspace(min(subset['alpha_scaled']), max(subset['alpha_scaled']), 100)
    y_pred = func_pow_ratio_alpha_fit_no(subset['alpha_scaled'], a)
    plt.scatter(subset['alpha_scaled'], subset['pow_ratio'], color=color, label=f'\u03F4 = {tita}')
    plt.plot(x, func_pow_ratio_alpha_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT={a}*x/(1+x)**2')
plt.axvline(x=1, color='red', linestyle='--', label='\u03B1')
avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1 scaled', fontsize=25)
plt.ylabel('Plat/P0', fontsize=25)
srednie_r2_alpha_fit = Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall eEffect \n Plat/P0 vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n No Offset \n FIT= a*x**2/(1+x**2) ', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()







###Graphing power vs Theta with offset - alpha or alpha_scaled? ##########CAN EXCHANGE ALPHA VS ALPHA_SCALED ################

colors_theta_fit = cm.rainbow(np.linspace(0, 1, len(Alphas_for_few_fits)))
r2_list_theta_fit=[]
plt.figure(figsize=(30,20))
for alpha, color in zip(Alphas_for_few_fits, colors_theta_fit):
    subset= df_HB[df_HB['alpha'] == alpha]
    popt, pcov, y_pred, r2 = fit_data(subset['Theta'], subset['pow_ratio'], func_pow_ratio_theta_fit, initial_guess=[1, 1])
    r2_list_theta_fit.append(r2)
    a, c = popt
    x = np.linspace(min(subset['Theta']), max(subset['Theta']), 100)
    y_pred = func_pow_ratio_theta_fit(subset['Theta'], a, c)
    plt.scatter(subset['Theta'], subset['pow_ratio'], color=color, label=f'\u03B1 = {alpha}')
    # Plot the fitted function with color
    plt.plot(x, func_pow_ratio_theta_fit(x, a, c), color=color, label= f'r2={r2},\n FIT={a}*x**2/(1+x**2) + {c}' )

avg_r2 = Average(r2_list_theta_fit)
plt.xlabel('\u03F4', fontsize=25)
plt.ylabel('Plat/P0', fontsize=25)
srednie_r2_theta_fit= Average(r2_list_theta_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Plat/P0 vs \u03B1 at given \u03F41, average r2 = {avg_r2} \n With Offset \n FIT= a*x**2/(1+x**2) + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

r2_list_theta_fit=[]
plt.figure(figsize=(30,20))
for alpha, color in zip(Alphas_for_few_fits, colors_theta_fit):
    subset= df_HB[df_HB['alpha'] == alpha]
    
    popt, pcov, y_pred, r2 = fit_data(subset['Theta'], subset['pow_ratio'], func_pow_ratio_theta_fit, initial_guess=[1, 1])
    a, c = popt
    
    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
    subset['Theta'], subset['pow_ratio'], func_pow_ratio_theta_fit, func_pow_ratio_theta_fit_no, initial_guess=[1, 1])
    r2_list_theta_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'pow_ratio'] = subset['pow_ratio'] - c
    
    x = np.linspace(min(subset['Theta']), max(subset['Theta']), 100)
    y_pred = func_pow_ratio_theta_fit_no(subset['Theta'], a)
    plt.scatter(subset['Theta'], subset['pow_ratio'], color=color, label=f'\u03B1 = {alpha}')
    # Plot the fitted function with color
    plt.plot(x, func_pow_ratio_theta_fit_no(x, a), color=color, label= f'r2={r2},\n FIT={a}*x**2/(1+x**2)' )
    
avg_r2 = Average(r2_list_theta_fit)
plt.xlabel('\u03F4', fontsize=25)
plt.ylabel('Plat/P0', fontsize=25)
srednie_r2_theta_fit= Average(r2_list_theta_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Plat/P0 vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n No Offset \n FIT= a*x**2/(1+x**2)', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

# %% Grpahing HB Voltage functions


###With Offset Hall Voltage vs alpha fit
r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset= df_HB[df_HB['Theta'] == tita]

    popt, pcov, y_pred, r2 = fit_data(subset['alpha'], subset['volt_dif'], func_volt_dif_alpha_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt

    x = np.linspace(min(subset['alpha']), max(subset['alpha']), 100)
    y_pred = func_volt_dif_alpha_fit(subset['alpha'], *popt)
    plt.scatter(subset['alpha'], subset['volt_dif'], color=color, label=f'\u03F4 = {tita}')
    # Plot the fitted function with color
    plt.plot(x, func_volt_dif_alpha_fit(x, a, c), color=color, label= f'r2={r2},\n FIT = {a}/(1+x) + {c}' )

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1', fontsize=25)
plt.ylabel('Hall Voltage [V]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Hall Voltage vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n With Offset \n FIT= a/(1+x) + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

###NO OFFSET Hall Voltage vs alpha fit

r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset = df_HB[df_HB['Theta'] == tita]
    
    popt, pcov, y_pred, r2 = fit_data(subset['alpha'], subset['volt_dif'], func_volt_dif_alpha_fit, initial_guess=[1, 1])
    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['alpha'], subset['volt_dif'], func_volt_dif_alpha_fit, func_volt_dif_alpha_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'volt_dif'] = subset['volt_dif'] - c
    
    x = np.linspace(min(subset['alpha']), max(subset['alpha']), 100)
    y_pred = func_volt_dif_alpha_fit_no(subset['alpha'], a)
    plt.scatter(subset['alpha'], subset['volt_dif'], color=color, label=f'\u03F4 = {tita}')
    plt.plot(x, func_volt_dif_alpha_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT = {a}/(1+x)')

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1', fontsize=25)
plt.ylabel('Hall Voltage [V]', fontsize=25)
srednie_r2_alpha_fit = Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Hall Voltage  vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n No Offset \n FIT= a/(1+x) ', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()



####Hall Voltage vs alpha_scaled with offset

###With Offset Hall Voltage vs alpha fit
r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset= df_HB[df_HB['Theta'] == tita]

    popt, pcov, y_pred, r2 = fit_data(subset['alpha_scaled'], subset['volt_dif'], func_volt_dif_alpha_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt

    x = np.linspace(min(subset['alpha_scaled']), max(subset['alpha_scaled']), 100)
    y_pred = func_volt_dif_alpha_fit(subset['alpha_scaled'], *popt)
    plt.scatter(subset['alpha_scaled'], subset['volt_dif'], color=color, label=f'\u03F4 = {tita}')
    # Plot the fitted function with color
    plt.plot(x, func_volt_dif_alpha_fit(x, a, c), color=color, label= f'r2={r2},\n FIT = {a}/(1+x) + {c}' )

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1 scaled', fontsize=25)
plt.ylabel('Hall Voltage [V]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Hall Voltage vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n With Offset \n FIT= a/(1+x) + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

###NO OFFSET Hall Voltage vs alpha_scaled fit

r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset = df_HB[df_HB['Theta'] == tita]
    
    popt, pcov, y_pred, r2 = fit_data(subset['alpha_scaled'], subset['volt_dif'], func_volt_dif_alpha_fit, initial_guess=[1, 1])
    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['alpha_scaled'], subset['volt_dif'], func_volt_dif_alpha_fit, func_volt_dif_alpha_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'volt_dif'] = subset['volt_dif'] - c
    
    x = np.linspace(min(subset['alpha_scaled']), max(subset['alpha_scaled']), 100)
    y_pred = func_volt_dif_alpha_fit_no(subset['alpha_scaled'], a)
    plt.scatter(subset['alpha_scaled'], subset['volt_dif'], color=color, label=f'\u03F4 = {tita}')
    plt.plot(x, func_volt_dif_alpha_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT = {a}/(1+x)')

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1 scaled', fontsize=25)
plt.ylabel('Hall Voltage [V]', fontsize=25)
srednie_r2_alpha_fit = Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Hall Voltage  vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n No Offset \n FIT= a/(1+x) ', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()




####Hall Voltage vs Theta

colors_theta_fit = cm.rainbow(np.linspace(0, 1, len(Alphas_for_few_fits)))
r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for alpha, color in zip(Alphas_for_few_fits, colors_theta_fit):
    subset= df_HB[df_HB['alpha'] == alpha]

    popt, pcov, y_pred, r2 = fit_data(subset['Theta'], subset['volt_dif'], func_volt_dif_theta_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt

    x = np.linspace(min(subset['Theta']), max(subset['Theta']), 100)
    y_pred = func_volt_dif_theta_fit(subset['Theta'], *popt)
    plt.scatter(subset['Theta'], subset['volt_dif'], color=color, label=f'\u03B1 = {alpha}')
    # Plot the fitted function with color
    plt.plot(x, func_volt_dif_theta_fit(x, a, c), color=color, label= f'r2={r2},\n FIT = {a}*x + {c}' )

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03F4', fontsize=25)
plt.ylabel('Hall Voltage [V]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Hall Voltage vs \u03F4 at given \u03B1, average r2 = {avg_r2} \n With Offset \n FIT= a*x + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

###NO OFFSET Hall Voltage vs alpha_scaled fit

colors_theta_fit = cm.rainbow(np.linspace(0, 1, len(Alphas_for_few_fits)))
r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for tita, color in zip(Alphas_for_few_fits, colors_theta_fit):
    subset = df_HB[df_HB['alpha'] == tita]
    
    popt, pcov, y_pred, r2 = fit_data(subset['Theta'], subset['volt_dif'], func_volt_dif_theta_fit, initial_guess=[1, 1])
    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['Theta'], subset['volt_dif'], func_volt_dif_theta_fit, func_volt_dif_theta_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'volt_dif'] = subset['volt_dif'] - c
    
    x = np.linspace(min(subset['Theta']), max(subset['Theta']), 100)
    y_pred = func_volt_dif_theta_fit_no(subset['Theta'], a)
    plt.scatter(subset['Theta'], subset['volt_dif'], color=color, label=f'\u03B1 = {alpha}')
    plt.plot(x, func_volt_dif_theta_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT = {a}*x ')

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03F4', fontsize=25)
plt.ylabel('Hall Voltage [V]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Hall Voltage vs \u03F4 at given \u03B1, average r2 = {avg_r2} \n No Offset \n FIT= a*x', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()


#%% FITTING CURRENT FUNCTIONS

###With Offset Current vs alpha fit
r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset= df_HB[df_HB['Theta'] == tita]

    popt, pcov, y_pred, r2 = fit_data(subset['alpha'], subset['avg_cur'], func_cur_y_alpha_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt

    x = np.linspace(min(subset['alpha']), max(subset['alpha']), 100)
    y_pred = func_cur_y_alpha_fit(subset['alpha'], *popt)
    plt.scatter(subset['alpha'], subset['avg_cur'], color=color, label=f'\u03F4 = {tita}')
    # Plot the fitted function with color
    plt.plot(x, func_cur_y_alpha_fit(x, a, c), color=color, label= f'r2={r2},\n FIT = {a}*x/(1+x) + {c}' )

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1', fontsize=25)
plt.ylabel('Load Current [A]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Load Current vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n With Offset \n FIT= a*x/(1+x) + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

###NO OFFSET Current vs alpha fit

r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset = df_HB[df_HB['Theta'] == tita]
    
    popt, pcov, y_pred, r2 = fit_data(subset['alpha'], subset['avg_cur'], func_cur_y_alpha_fit, initial_guess=[1, 1])
    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['alpha'], subset['avg_cur'], func_cur_y_alpha_fit, func_cur_y_alpha_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'avg_cur'] = subset['avg_cur'] - c
    
    x = np.linspace(min(subset['alpha']), max(subset['alpha']), 100)
    y_pred = func_cur_y_alpha_fit_no(subset['alpha'], a)
    plt.scatter(subset['alpha'], subset['avg_cur'], color=color, label=f'\u03F4 = {tita}')
    plt.plot(x, func_cur_y_alpha_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT = FIT = {a}*x/(1+x)')

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1', fontsize=25)
plt.ylabel('Load Current [A]', fontsize=25)
srednie_r2_alpha_fit = Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Load Current  vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n No Offset \n FIT= a*x/(1+x) ', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()



####Current Load vs alpha_scaled with offset

r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset= df_HB[df_HB['Theta'] == tita]

    popt, pcov, y_pred, r2 = fit_data(subset['alpha_scaled'], subset['avg_cur'], func_cur_y_alpha_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt

    x = np.linspace(min(subset['alpha_scaled']), max(subset['alpha_scaled']), 100)
    y_pred = func_cur_y_alpha_fit(subset['alpha_scaled'], *popt)
    plt.scatter(subset['alpha_scaled'], subset['avg_cur'], color=color, label=f'\u03F4 = {tita}')
    # Plot the fitted function with color
    plt.plot(x, func_cur_y_alpha_fit(x, a, c), color=color, label= f'r2={r2},\n FIT = {a}*x/(1+x) + {c}' )

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1 scaled', fontsize=25)
plt.ylabel('Load Current [A]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Load Current vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n With Offset \n FIT= a*x/(1+x) + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

###NO OFFSET Current load vs alpha_scaled fit

r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset = df_HB[df_HB['Theta'] == tita]
    
    popt, pcov, y_pred, r2 = fit_data(subset['alpha_scaled'], subset['avg_cur'], func_cur_y_alpha_fit, initial_guess=[1, 1])
    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['alpha_scaled'], subset['avg_cur'], func_cur_y_alpha_fit, func_cur_y_alpha_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'avg_cur'] = subset['avg_cur'] - c
    
    x = np.linspace(min(subset['alpha_scaled']), max(subset['alpha_scaled']), 100)
    y_pred = func_cur_y_alpha_fit_no(subset['alpha'], a)
    plt.scatter(subset['alpha_scaled'], subset['avg_cur'], color=color, label=f'\u03F4 = {tita}')
    plt.plot(x, func_cur_y_alpha_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT = FIT = {a}*x/(1+x)')

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1 scaled', fontsize=25)
plt.ylabel('Load Current [A]', fontsize=25)
srednie_r2_alpha_fit = Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Load Current  vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n No Offset \n FIT= a*x/(1+x) ', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()




####Current Load vs Theta

colors_theta_fit = cm.rainbow(np.linspace(0, 1, len(Alphas_for_few_fits)))
r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for alpha, color in zip(Alphas_for_few_fits, colors_theta_fit):
    subset= df_HB[df_HB['alpha'] == alpha]

    popt, pcov, y_pred, r2 = fit_data(subset['Theta'], subset['avg_cur'], func_cur_y_theta_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt

    x = np.linspace(min(subset['Theta']), max(subset['Theta']), 100)
    y_pred = func_cur_y_theta_fit(subset['Theta'], *popt)
    plt.scatter(subset['Theta'], subset['avg_cur'], color=color, label=f'\u03B1 = {alpha}')
    # Plot the fitted function with color
    plt.plot(x, func_cur_y_theta_fit(x, a, c), color=color, label= f'r2={r2},\n FIT = {a}*x + {c}' )

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03F4', fontsize=25)
plt.ylabel('Load Current [A]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Hall Voltage vs \u03F4 at given \u03B1, average r2 = {avg_r2} \n With Offset \n FIT= a*x + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

###NO OFFSET Current vs alpha_scaled fit

colors_theta_fit = cm.rainbow(np.linspace(0, 1, len(Alphas_for_few_fits)))
r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for tita, color in zip(Alphas_for_few_fits, colors_theta_fit):
    subset = df_HB[df_HB['alpha'] == tita]
    
    popt, pcov, y_pred, r2 = fit_data(subset['Theta'], subset['avg_cur'], func_cur_y_theta_fit, initial_guess=[1, 1])
    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['Theta'], subset['avg_cur'], func_cur_y_theta_fit, func_cur_y_theta_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'avg_cur'] = subset['avg_cur'] - c
    
    x = np.linspace(min(subset['Theta']), max(subset['Theta']), 100)
    y_pred = func_cur_y_theta_fit_no(subset['Theta'], a)
    plt.scatter(subset['Theta'], subset['avg_cur'], color=color, label=f'\u03B1 = {alpha}')
    plt.plot(x, func_cur_y_theta_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT = {a}*x ')

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03F4', fontsize=25)
plt.ylabel('Load Current [A]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Hall Voltage vs \u03F4 at given \u03B1, average r2 = {avg_r2} \n No Offset \n FIT= a*x', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()




#%% GRAPHING CHARGE ACC:
    

r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset= df_HB[df_HB['Theta'] == tita]

    popt, pcov, y_pred, r2 = fit_data(subset['alpha'], subset['charge_acc'], func_charge_alpha_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt

    x = np.linspace(min(subset['alpha']), max(subset['alpha']), 100)
    y_pred = func_charge_alpha_fit(subset['alpha'], *popt)
    plt.scatter(subset['alpha'], subset['charge_acc'], color=color, label=f'\u03F4 = {tita}')
    # Plot the fitted function with color
    plt.plot(x, func_charge_alpha_fit(x, a, c), color=color, label= f'r2={r2},\n FIT = {a}*x/(1+x) + {c}' )

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1', fontsize=25)
plt.ylabel('Charge Acumulation on the insulating boundary [C/m]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Charge Acumulation vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n With Offset \n FIT= a*x/(1+x) + c', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

###NO OFFSET Charge Acc vs alpha fit

r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset = df_HB[df_HB['Theta'] == tita]
    
    popt, pcov, y_pred, r2 = fit_data(subset['alpha'], subset['charge_acc'], func_charge_alpha_fit, initial_guess=[1, 1])
    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['alpha'], subset['charge_acc'], func_charge_alpha_fit, func_charge_alpha_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'charge_acc'] = subset['charge_acc'] - c
    
    x = np.linspace(min(subset['alpha']), max(subset['alpha']), 100)
    y_pred = func_charge_alpha_fit_no(subset['alpha'], a)
    plt.scatter(subset['alpha'], subset['charge_acc'], color=color, label=f'\u03F4 = {tita}')
    plt.plot(x, func_charge_alpha_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT = FIT = {a}*x/(1+x)')

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1', fontsize=25)
plt.ylabel('Charge Acumulation on the insulating boundary [C/m]', fontsize=25)
srednie_r2_alpha_fit = Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Charge Acumulation  vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n No Offset \n FIT= a*x/(1+x) ', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()



####Charge Acc vs alpha_scaled with offset

r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset= df_HB[df_HB['Theta'] == tita]

    popt, pcov, y_pred, r2 = fit_data(subset['alpha_scaled'], subset['charge_acc'], func_charge_alpha_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt

    x = np.linspace(min(subset['alpha_scaled']), max(subset['alpha_scaled']), 100)
    y_pred = func_charge_alpha_fit(subset['alpha_scaled'], *popt)
    plt.scatter(subset['alpha_scaled'], subset['charge_acc'], color=color, label=f'\u03F4 = {tita}')
    # Plot the fitted function with color
    plt.plot(x, func_charge_alpha_fit(x, a, c), color=color, label= f'r2={r2},\n FIT = {a}*x/(1+x) + {c}' )

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1 scaled', fontsize=25)
plt.ylabel('Charge Acumulation on the insulating boundary [C/m]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Charge Acumulation vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n With Offset \n FIT= a*x/(1+x) + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

###NO OFFSET Charge Acc vs alpha_scaled fit

r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for tita, color in zip(Thetas_for_few_fits, colors_alpha_fit):
    subset = df_HB[df_HB['Theta'] == tita]
    
    popt, pcov, y_pred, r2 = fit_data(subset['alpha_scaled'], subset['charge_acc'], func_cur_y_alpha_fit, initial_guess=[1, 1])
    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['alpha_scaled'], subset['charge_acc'], func_charge_alpha_fit, func_charge_alpha_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'charge_acc'] = subset['charge_acc'] - c
    
    x = np.linspace(min(subset['alpha_scaled']), max(subset['alpha_scaled']), 100)
    y_pred = func_charge_alpha_fit_no(subset['alpha_scaled'], a)
    plt.scatter(subset['alpha_scaled'], subset['charge_acc'], color=color, label=f'\u03F4 = {tita}')
    plt.plot(x, func_charge_alpha_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT = FIT = {a}*x/(1+x)')

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03B1 scaled', fontsize=25)
plt.ylabel('Charge Acumulation on the insulating boundary [C/m]', fontsize=25)
srednie_r2_alpha_fit = Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Charge Acumulation  vs \u03B1 at given \u03F4, average r2 = {avg_r2} \n No Offset \n FIT= a*x/(1+x) ', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()




####Charge Acc vs Theta

colors_theta_fit = cm.rainbow(np.linspace(0, 1, len(Alphas_for_few_fits)))
r2_list_alpha_fit=[]
plt.figure(figsize=(30,20))
for alpha, color in zip(Alphas_for_few_fits, colors_theta_fit):
    subset= df_HB[df_HB['alpha'] == alpha]

    popt, pcov, y_pred, r2 = fit_data(subset['Theta'], subset['charge_acc'], func_charge_theta_fit, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2)
    a, c = popt

    x = np.linspace(min(subset['Theta']), max(subset['Theta']), 100)
    y_pred = func_charge_theta_fit(subset['Theta'], *popt)
    plt.scatter(subset['Theta'], subset['charge_acc'], color=color, label=f'\u03B1 = {alpha}')
    # Plot the fitted function with color
    plt.plot(x, func_charge_theta_fit(x, a, c), color=color, label= f'r2={r2},\n FIT = {a}*x/(1+x**2) + {c}' )

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03F4', fontsize=25)
plt.ylabel('Charge Acumulation on the insulating boundary [C/m]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Charge Acumulation vs \u03F4 at given \u03B1, average r2 = {avg_r2} \n With Offset \n FIT= a*x/(1+x**2) + c', fontsize=30)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()

###NO OFFSET Charge acc vs theta fit

colors_theta_fit = cm.rainbow(np.linspace(0, 1, len(Alphas_for_few_fits)))
r2_list_alpha_fit = []
plt.figure(figsize=(30,20))
for alpha, color in zip(Alphas_for_few_fits, colors_theta_fit):
    subset = df_HB[df_HB['alpha'] == alpha]
    
    popt, pcov, y_pred, r2 = fit_data(subset['Theta'], subset['charge_acc'], func_charge_theta_fit, initial_guess=[1, 1])
    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
        subset['Theta'], subset['charge_acc'], func_charge_theta_fit, func_charge_theta_fit_no, initial_guess=[1, 1])
    r2_list_alpha_fit.append(r2_no)
    a = popt_no[0]
    
    subset.loc[:, 'charge_acc'] = subset['charge_acc'] - c
    
    x = np.linspace(min(subset['Theta']), max(subset['Theta']), 100)
    y_pred = func_charge_theta_fit_no(subset['Theta'], a)
    plt.scatter(subset['Theta'], subset['charge_acc'], color=color, label=f'\u03B1 = {alpha}')
    plt.plot(x, func_charge_theta_fit_no(x, a), color=color,label=f'r2={r2_no},\n FIT = {a}*x/(1+x**2) ')

avg_r2 = Average(r2_list_alpha_fit)
plt.xlabel('\u03F4', fontsize=25)
plt.ylabel('Charge Acumulation on the insulating boundary [C/m]', fontsize=25)
srednie_r2_alpha_fit= Average(r2_list_alpha_fit)
plt.title(f'Hall Bar Simulation Results for Anomalous Hall effect \n Charge Acumulation vs \u03F4 at given \u03B1, average r2 = {avg_r2} \n No Offset \n FIT= a*x/(1+x**2)', fontsize=30)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)  # Moving legend below the plot
plt.show()
plt.clf()


