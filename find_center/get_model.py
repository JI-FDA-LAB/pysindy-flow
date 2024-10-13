import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp

import pysindy as ps

import os

import math  
import sys  
sys.path.append('C:/Users/j/hotai/myfolder')

from my_function import miscore, siscore
# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Seed the random number generators for reproducibility
np.random.seed(100)

def getmodel(v_train, spatial_grid, dt_train):
    
    
    v_train_dot = ps.FiniteDifference(axis=2)._differentiate(v_train, dt_train)
    # Define PDE library that is quadratic in u, and
    # fourth-order in spatial derivatives of u.    periodic=True

    library_functions = [
        lambda y: y,
        lambda x: x*x,
        
        lambda x, y: x * y,
        lambda x, y: x * x * y,
        lambda x , y , z: x * y * z
    ]
    library_function_names = [
        lambda y: y,
        lambda x: x+x,
        
        lambda x, y: x + y,
        lambda x, y: x + x + y,
        lambda x , y , z: x + y + z
    ]
    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=1,
        spatial_grid=spatial_grid,
        include_bias=True,
        is_uniform=True
        
    )
    print('SINDy')
    model = ps.SINDy(feature_library=pde_lib, feature_names=['Vx','Vy','t'])

    model.fit(v_train, x_dot=v_train_dot)
    model.print()

    print("Model score: %f" % model.score(v_train, t=dt_train))

    return model



def ori_di_test(v_train, t_train):
    dt_train=t_train[1]-t_train[0]
    v_train_dot = ps.FiniteDifference(axis=2)._differentiate(v_train, dt_train)

    vtmp=np.zeros(v_train_dot.shape)
    vtmp[:,:,0]=v_train[:,:,0]
    for i in range(len(t_train)-1):
        vtmp[:,:,i+1]=vtmp[:,:,i]+v_train_dot[:,:,i]*dt_train
    #miscore(vtmp,v_train)
    #siscore(vtmp,v_train)
    vtmp=v_train+v_train_dot*dt_train

    vtmp[:,:,1:]=vtmp[:,:,:len(t_train)-1]
    #miscore(vtmp,v_train)
    #siscore(vtmp,v_train)