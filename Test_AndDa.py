# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:52:40 2020

@author: hirvoasa
"""
import pylab
import numpy as np
import matplotlib.pyplot as plt
pylab.rcParams['figure.figsize'] = (16, 9)

# analog data assimilation
from AnDA_codes.AnDA_generate_data import AnDA_generate_data
from AnDA_codes.AnDA_analog_forecasting import AnDA_analog_forecasting
from AnDA_codes.AnDA_model_forecasting import AnDA_model_forecasting
from AnDA_codes.AnDA_data_assimilation import AnDA_data_assimilation
from AnDA_codes.AnDA_stat_functions import AnDA_RMSE

# 3D plots
from mpl_toolkits.mplot3d import Axes3D

### GENERATE SIMULATED DATA (LORENZ-63 MODEL)

# parameters
class GD:
    model = 'Lorenz_63'
    class parameters:
        sigma = 10.0
        rho = 28.0
        beta = 8.0/3
    dt_integration = 0.01 # integration time
    dt_states = 1 # number of integeration times between consecutive states (for xt and catalog)
    dt_obs = 8 # number of integration times between consecutive observations (for yo)
    var_obs = np.array([0]) # indices of the observed variables
    nb_loop_train = 10**2 # size of the catalog
    nb_loop_test = 10 # size of the true state and noisy observations
    sigma2_catalog = 0.0 # variance of the model error to generate the catalog
    sigma2_obs = 2.0 # variance of the observation error to generate observation
    
# run the data generation
catalog, xt, yo = AnDA_generate_data(GD)


### PLOT STATE, OBSERVATIONS AND CATALOG

# state and observations (when available)
line1,=plt.plot(xt.time,xt.values[:,0],'-r');plt.plot(yo.time,yo.values[:,0],'.r')
line2,=plt.plot(xt.time,xt.values[:,1],'-b');plt.plot(yo.time,yo.values[:,1],'.b')
line3,=plt.plot(xt.time,xt.values[:,2],'-g');plt.plot(yo.time,yo.values[:,2],'.g')
plt.xlabel('Lorenz-63 times')
plt.legend([line1, line2, line3], ['$x_1$', '$x_2$', '$x_3$'])
plt.title('Lorenz-63 true (continuous lines) and observed trajectories (points)')

# catalog sample of simulated trajectories
print('######################')
print('SAMPLE OF THE CATALOG:')
print('######################')
print('Analog (t1): ' + str(catalog.analogs[0,:]), end=" ")
print('Successor (t1+dt): ' + str(catalog.successors[0,:]))
print('Analog (t2): ' + str(catalog.analogs[10,:]), end=" ")
print('Successor (t2+dt): ' + str(catalog.successors[10,:]))
print('Analog (t3): ' + str(catalog.analogs[20,:]), end=" ")
print('Successor (t3+dt): ' + str(catalog.successors[20,:]))

### ANALOG DATA ASSIMILATION (dynamical model given by the catalog)

# parameters of the analog forecasting method
class AF:
    k = 50 # number of analogs
    neighborhood = np.ones([xt.values.shape[1],xt.values.shape[1]]) # global analogs
    catalog = catalog # catalog with analogs and successors
    regression = 'locally_constant' # chosen regression ('locally_constant', 'increment', 'local_linear')
    sampling = 'gaussian' # chosen sampler ('gaussian', 'multinomial')

# parameters of the filtering method
class DA:
    method = 'AnEnKS' # chosen method ('AnEnKF', 'AnEnKS', 'AnPF')
    N = 100 # number of members (AnEnKF/AnEnKS) or particles (AnPF)
    xb = xt.values[0,:]; B = 0.1*np.eye(xt.values.shape[1])
    H = np.eye(xt.values.shape[1])
    R = GD.sigma2_obs*np.eye(xt.values.shape[1])
    @staticmethod
    def m(x):
        return AnDA_analog_forecasting(x,AF)
    
# run the analog data assimilation
x_hat_analog = AnDA_data_assimilation(yo, DA)


### CLASSICAL DATA ASSIMILATION (dynamical model given by the equations)
    
# parameters of the filtering method
class DA:
    method = 'AnEnKS' # chosen method ('AnEnKF', 'AnEnKS', 'AnPF')
    N = 100 # number of members (AnEnKF/AnEnKS) or particles (AnPF)
    xb = xt.values[0,:]; B = 0.1*np.eye(xt.values.shape[1])
    H = np.eye(xt.values.shape[1])
    R = GD.sigma2_obs*np.eye(xt.values.shape[1])
    @staticmethod
    def m(x):
        return AnDA_model_forecasting(x,GD)
    
# run the classical data assimilation
x_hat_classical = AnDA_data_assimilation(yo, DA)

### COMPARISON BETWEEN CLASSICAL AND ANALOG DATA ASSIMILATION

# plot
fig=plt.figure()
ax=fig.gca(projection='3d')
line1,=ax.plot(xt.values[:,0],xt.values[:,1],xt.values[:,2],'k')
line2,=ax.plot(x_hat_analog.values[:,0],x_hat_analog.values[:,1],x_hat_analog.values[:,2],'r')
line3,=ax.plot(x_hat_classical.values[:,0],x_hat_classical.values[:,1],x_hat_classical.values[:,2],'b')
ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$');ax.set_zlabel('$x_3$')
plt.legend([line1, line2, line3], ['True state', 'Analog data assimilation', 'Classical data assimilation'])
plt.title('Reconstruction of Lorenz-63 trajectories')

# error
print('RMSE(analog DA)    = ' + str(AnDA_RMSE(xt.values,x_hat_analog.values)))
print('RMSE(classical DA) = ' + str(AnDA_RMSE(xt.values,x_hat_classical.values)))