# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import pygmo
import numpy as np
from scipy import interpolate, integrate, optimize      # used to integrate weights numerically
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import multiprocessing as mp

min_pos_system_value = (np.finfo(float).tiny*(1E20))**(1/2)
max_pos_system_value = (np.finfo(float).max*(1E-20))**(1/2)

def generalized_loss_fcn(x, mu=0, a=2, c=1):    # defaults to L2 loss
    x_c_2 = ((x-mu)/c)**2
    
    if a == 1:          # generalized function reproduces
        loss = (x_c_2 + 1)**(0.5) - 1
    if a == 2:
        loss = 0.5*x_c_2
    elif a == 0:
        loss = np.log(0.5*x_c_2+1)
    elif a == -2:       # generalized function reproduces
        loss = 2*x_c_2/(x_c_2 + 4)
    elif a <= -1000:    # supposed to be negative infinity
        loss = 1 - np.exp(-0.5*x_c_2)
    else:
        loss = np.abs(a-2)/a*((x_c_2/np.abs(a-2) + 1)**(a/2) - 1)

    return loss*c**a + mu  # multiplying by c^2 is not necessary, but makes order appropriate

def sig(x):     # Numerically stable sigmoid function
    eval = np.empty_like(x)
    pos_val_f = np.exp(-x[x >= 0])
    eval[x >= 0] = 1/(1 + pos_val_f)
    neg_val_f = np.exp(x[x < 0])
    eval[x < 0] = neg_val_f/(1 + neg_val_f)

    # clip so they can be multiplied
    eval[eval > 0] = np.clip(eval[eval > 0], min_pos_system_value, max_pos_system_value)
    eval[eval < 0] = np.clip(eval[eval < 0], -max_pos_system_value, -min_pos_system_value)

    return eval

def sig_deriv(sig_val):
    return sig_val*(1-sig_val)
  
def eval_coef(tau, a):
    return a[0] + a[1]*np.exp(a[2]*tau)
  
def Z_fun(tau, alpha, *vars):
    x = np.log(-alpha + 2 + 1/1001)
    
    L_x0 = eval_coef(tau, vars[0:3])
    k = eval_coef(tau, vars[3:6])
    A1 = [eval_coef(tau, vars[6:9]), eval_coef(tau, vars[9:12])]
    A2 = [eval_coef(tau, vars[12:15]), eval_coef(tau, vars[15:18])]
    p  = [eval_coef(tau, vars[18:21]), eval_coef(tau, vars[21:24]), eval_coef(tau, vars[24:27])]
    x0 = [eval_coef(tau, vars[27:30]), eval_coef(tau, vars[30:33]), eval_coef(tau, vars[33:36]), 
          eval_coef(tau, vars[36:39]), eval_coef(tau, vars[39:42])]
    h  = [eval_coef(tau, vars[42:45]), eval_coef(tau, vars[45:48]), eval_coef(tau, vars[48:51]), 
          eval_coef(tau, vars[51:54]), eval_coef(tau, vars[54:57])]
    
    p = [p[0], 1-p[0], p[1], p[2], 1-p[1]-p[2]]
    
    L = sig((x - L_x0)/k)
    f_1 = A1[0]+(A2[0]-A1[0])*(p[0]/(1+10**((x0[0]-x)*h[0])) + p[1]/(1+10**((x0[1]-x)*h[1]))+ p[2]/(1+10**((x0[2]-x)*h[2])))
    f_2 = A1[1]+(A2[1]-A1[1])*(p[3]/(1+10**((x0[3]-x)*h[3])) + p[4]/(1+10**((x0[4]-x)*h[4])))
    
    return np.exp(L*f_2 + (1-L)*f_1)
    
class pygmo_objective_fcn:
    def __init__(self, bnds, tau, alpha, Z):
        self.bnds = bnds
        self.tau = np.unique(tau)
        self.alpha = np.unique(alpha)
        self.Z = Z
        self.ln_Z = np.log(Z)

    def fitness(self, x):
        obj = 0
        for i in range(len(self.tau)):
            # print(self.tau[i], self.alpha[i], Z_fun(self.tau[i], self.alpha[i], *x))
            try:
                Z_fit = Z_fun(self.tau[i], self.alpha[i], *x)
                if np.isnumeric(Z_fit):
                    obj += 100*(np.log(Z_fit) - self.ln_Z[i])**2
                    print(obj)
                else:
                    obj += 1
            except:
                obj += 1
            
        return [obj]

    def get_bounds(self):
        return self.bnds

    def gradient(self, x):
        return pygmo.estimate_gradient_h(lambda x: self.fitness(x), x)
   

'''
alpha_vec = np.linspace(-1000, -10, 991)
alpha_vec = np.concatenate([alpha_vec, np.linspace(-10, 2, 1201)[1:]])
tau = 250
res = []
c = 1
for tau in np.linspace(1, 250, 250):
    for alpha in alpha_vec:
        int_func = lambda x: np.exp(-generalized_loss_fcn(x, a=alpha, c=1))
        Z, err = integrate.quadrature(int_func, -tau, tau, tol=1E-14, maxiter=10000)

        res.append([tau, c, alpha, Z])
        print(f'{tau:0.3f} {alpha:0.3f} {Z:0.14e}')
        
res = np.array(res)
np.savetxt('res.csv', res, delimiter=',')
'''

data = np.genfromtxt('res.csv', delimiter=',')

data = data[:, [0,2,3]]
tau = data[:,0].reshape((250,2191))
alpha = data[:,1].reshape((250,2191))
Z = data[:,2].reshape((250,2191))

tau = tau[:,1:]
alpha = alpha[:,1:]
Z = Z[:,1:]

kx = 3
ky = kx

# spline_fit = interpolate.RectBivariateSpline(tau[:,0], alpha[0,:], np.log(Z), kx=kx, ky=ky, s=0.0001)

# with open("spline_save.csv", "ab") as f:
    # for i in range(3):
        # np.savetxt(f, np.matrix(spline_fit.tck[i]), delimiter=",")
    # for val in [kx, ky]:
        # np.savetxt(f, np.matrix(val), delimiter=",")

tck = []
with open("spline_save.csv") as f:
    for i in range(5):
        tck.append(np.array(f.readline().split(','), dtype=float))        

spline_fit = interpolate.RectBivariateSpline._from_tck(tck)

idx = np.argwhere(data[:,0] == 1)

# plt.plot(x_deriv, deriv(x_deriv))
# plt.plot(data[idx,1], np.log(data[idx,2]) - spline_fit([2], data[idx,1]).T)
# plt.plot(data[idx,1][1:], np.log(data[idx,2][1:]) - spline_fit([2], data[idx,1][1:]).T)
plt.plot(data[idx,1][1:], np.log(data[idx,2][1:]), data[idx,1][1:], spline_fit([1], data[idx,1][1:]).T)
plt.show()