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



###path_HB_Ideal = Specify the path to the file that is being read



hall_resistance_HB_Ideal = 121.21212 #Value of the resistance of the Hall's Bar from simulation


current_device = 0.00053


# %% Making of df_HB_Ideal:

df_HB_Ideal = pd.DataFrame(columns=[ 'Theta', 'pow_0', 'volt_dif', 'charge_acc'])
# charge_acc is actually the difference in charge accumulation on the two sides

list_of_thetas_for_columns_HB_Ideal = []


# READING OF data HB_Ideal
with open(path_HB_Ideal, 'r') as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if the line starts with '% Data (P0 (W) @'
        if line.startswith('% Data (pow_0 @'):
            tita = line.split('Theta=')[1].split(',')[0].replace(")", "")
            theta = float(tita)
            list_of_thetas_for_columns_HB_Ideal.append(theta)
        i = i+1

list_of_thetas_for_columns_HB_Ideal_unique = list(OrderedSet(
    list_of_thetas_for_columns_HB_Ideal))  # LOOKING FOR UNIQUE THETAS HB_Ideal



df_temp_HB_Ideal = pd.DataFrame(list_of_thetas_for_columns_HB_Ideal_unique, columns=['Theta'])
df_HB_Ideal[['Theta']] = df_temp_HB_Ideal


# POPULATING df_HB_Ideal
with open(path_HB_Ideal, 'r') as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if the line starts with '% Data (P0 (W) @'
        if line.startswith('% Data (pow_0 @'):
            tita = line.split('Theta=')[1].split(',')[0].replace(")", "")
            theta = float(tita)
            pow_0_value = float(lines[i+1].strip())
            df_HB_Ideal.loc[(df_HB_Ideal['Theta'] == theta), 'pow_0'] = pow_0_value

        elif line.startswith('% Data (volt_up - volt_down @'):
            tita = line.split('Theta=')[1].split(',')[0].replace(")", "")
            theta = float(tita)
            volt_dif_value = -float(lines[i+1].strip())
            df_HB_Ideal.loc[(df_HB_Ideal['Theta'] == theta) , 'volt_dif'] = volt_dif_value

        elif line.startswith('% Data (charge_up - charge_down @'):
            tita = line.split('Theta=')[1].split(',')[0].replace(")", "")
            theta = float(tita)
            charge_acc_value = -(float(lines[i+1].strip()))
            df_HB_Ideal.loc[(df_HB_Ideal['Theta'] == theta) , 'charge_acc'] = charge_acc_value
        i = i+1

###FUNCTIONS TO DO THE FIT:
    
def func_volt_charge_fit(x,a,c):
    return a*x/(1+x**2) + c

def func_volt_charge_fit_no(x,a):
    return a*x/(1+x**2) 

def func(x, b):
    return np.full_like(x, b)

b = np.mean(df_HB_Ideal['pow_0'])
#%% data fits
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


#%% FITTING


###FITTING VOLT DIF VS THETA WITH OFFSET
i=0
plt.figure(figsize=(30,20))
while i<1:
    popt, pcov, y_pred, r2 = fit_data(df_HB_Ideal['Theta'], df_HB_Ideal['volt_dif'], func_volt_charge_fit, initial_guess=[1, 1])

    a, c = popt

    x = np.linspace(min(df_HB_Ideal['Theta']), max(df_HB_Ideal['Theta']), 100)
    y_pred = func_volt_charge_fit(df_HB_Ideal['Theta'], *popt)
    plt.scatter(df_HB_Ideal['Theta'], df_HB_Ideal['volt_dif'], color='blue', label='')
    plt.plot(x, func_volt_charge_fit(x, a, c), color='red', label= '' )
    plt.xlabel('\u03F4', fontsize=25)
    plt.ylabel('Hall Voltage [V]', fontsize=25)
    
    i=i+1
plt.title(f'Ideal Hall Bar Simulation Results for Anomalous Hall effect \n r2 = {r2:.9f} \n With Offset \n FIT={a}*x/(1+x**2) + {c} ', fontsize=30)
plt.show()
plt.clf()

i=0
plt.figure(figsize=(30,20))
while i<1:
    popt, pcov, y_pred, r2 = fit_data(df_HB_Ideal['Theta'], df_HB_Ideal['volt_dif'], func_volt_charge_fit, initial_guess=[1, 1])

    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
    df_HB_Ideal['Theta'], df_HB_Ideal['volt_dif'], func_volt_charge_fit, func_volt_charge_fit_no, initial_guess=[1, 1])

    a = popt_no[0]
    df_HB_Ideal.loc[:, 'volt_dif'] = df_HB_Ideal['volt_dif'] - c
    x = np.linspace(min(df_HB_Ideal['Theta']), max(df_HB_Ideal['Theta']), 100)
    y_pred = func_volt_charge_fit_no(df_HB_Ideal['Theta'], a)
    plt.scatter(df_HB_Ideal['Theta'], df_HB_Ideal['volt_dif'], color='blue', label='')
    plt.plot(x, func_volt_charge_fit_no(x, a), color='red', label= '' )
    plt.xlabel('\u03F4', fontsize=25)
    plt.ylabel('Hall Voltage [V]', fontsize=25)
    
    i=i+1
plt.title(f'Ideal Hall Bar Simulation Results for Anomalous Hall effect \n r2 = {r2:.9f} \n No Offset \n FIT={a}*x/(1+x**2) ', fontsize=30)
plt.show()
plt.clf()




i=0
plt.figure(figsize=(30,20))
###FITTING Charge acc VS THETA WITH OFFSET
while i<1:
    popt, pcov, y_pred, r2 = fit_data(df_HB_Ideal['Theta'], df_HB_Ideal['charge_acc'], func_volt_charge_fit, initial_guess=[1, 1])
    
    a, c = popt

    x = np.linspace(min(df_HB_Ideal['Theta']), max(df_HB_Ideal['Theta']), 100)
    y_pred = func_volt_charge_fit(df_HB_Ideal['Theta'], *popt)
    plt.scatter(df_HB_Ideal['Theta'], df_HB_Ideal['charge_acc'], color='blue', label='')

    plt.plot(x, func_volt_charge_fit(x, a, c), color='red', label= '' )
    plt.xlabel('\u03F4', fontsize=25)
    plt.ylabel('Charge Acumulation on the insulating boundary [C/m]', fontsize=25)
    i=i+1
plt.title(f'Ideal Hall Bar Simulation Results for Anomalous Hall effect \n r2 = {r2:.9f} \n With Offset \n FIT={a}*x/(1+x**2) + {c} ', fontsize=30)
plt.show()
plt.clf()

i=0
i=0
plt.figure(figsize=(30,20))
while i<1:
    popt, pcov, y_pred, r2 = fit_data(df_HB_Ideal['Theta'], df_HB_Ideal['charge_acc'], func_volt_charge_fit, initial_guess=[1, 1])

    a, c = popt

    popt_no, pcov_no, y_pred_no, r2_no = fit_data_no(
    df_HB_Ideal['Theta'], df_HB_Ideal['charge_acc'], func_volt_charge_fit, func_volt_charge_fit_no, initial_guess=[1, 1])

    a = popt_no[0]
    df_HB_Ideal.loc[:, 'charge_acc'] = df_HB_Ideal['charge_acc'] - c
    x = np.linspace(min(df_HB_Ideal['Theta']), max(df_HB_Ideal['Theta']), 100)
    y_pred = func_volt_charge_fit_no(df_HB_Ideal['Theta'], a)
    plt.scatter(df_HB_Ideal['Theta'], df_HB_Ideal['charge_acc'], color='blue', label='')
    plt.plot(x, func_volt_charge_fit_no(x, a), color='red', label= '' )
    plt.xlabel('\u03F4', fontsize=25)
    plt.ylabel('Charge Acumulation on the insulating boundary [C/m]', fontsize=25)
    
    i=i+1
plt.title(f'Ideal Hall Bar Simulation Results for Anomalous Hall effect \n r2 = {r2:.9f} \n No Offset \n FIT={a}*x/(1+x**2) ', fontsize=30)
plt.show()
plt.clf()
i=0



# ####FITTING pow_0 vs theta:
# while i<1:
#     plt.figure(figsize=(30,20))
#     y_true = df_HB_Ideal['pow_0']
#     y_pred = func(df_HB_Ideal['Theta'], b)
#     r2 = r2_score(y_true, y_pred)
#     x = np.linspace(min(df_HB_Ideal['Theta']), max(df_HB_Ideal['Theta']), 100)
#     plt.scatter(df_HB_Ideal['Theta'], df_HB_Ideal['pow_0'], color='blue', label='Data')
    
#     plt.xlabel('\u03F4', fontsize=25)
#     plt.ylabel('Power on Ideal Hall Bar [W]', fontsize=25)
#     plt.title('Ideal Hall Bar Simulation Results for Anomalous Hall effect ', fontsize=30)
#     plt.legend()
#     plt.show()
#     plt.clf()    
    
#     i=i+1    
# i=0


    


