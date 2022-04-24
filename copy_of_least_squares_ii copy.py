# -*- coding: utf-8 -*-

# Do not edit this cell.

LabID="Lab7"

try:
  from graderHelp import ISGRADEPLOT
except ImportError:
  ISGRADEPLOT = True


# Enter your first and last names in between the quotation marks.

first_name="Drew"

last_name="Wilson"

# Enter your Math 215 section number in between the quotation marks. 

section_number="Your Math 215 section number goes here"  

# Enter your BYU NetID in between the quotation marks.  NOT YOUR BYU ID NUMBER! 

BYUNetID="aw053102"


import numpy as np
import pandas as pd

df = pd.read_csv('Lab7data.csv')
signal_data=df.values
signal_data


T= signal_data[:,0]

Y= signal_data[:,1]

print(T.shape, Y.shape)

import matplotlib.pyplot as plt

plt.plot(T, Y)
plt.show()

def row_func(t,n):
  l = [f(k*t) for k in range(1,n + 1) for f in [np.cos,np.sin]]
  l.insert(0, 1)
  return l

l = row_func(2,5)

def design_matrix(n):
  M = [row_func(T[i], n) for i in range(len(T))]
  return np.array(M)

M = design_matrix(4)
print(M[100,:])

X2= design_matrix(2)

normal_coef2= np.matmul(np.transpose(X2), X2)

normal_vect2= np.matmul(np.transpose(X2), Y)

beta2= np.linalg.solve(normal_coef2, normal_vect2)

def f2(t):
  a = np.array(row_func(t, 2))
  return np.dot(beta2, a)

f2(.75)
 
vf2=np.vectorize(f2)    

plt.plot(T,Y,'r.')      

plt.plot(T,vf2(T),'b-') 

plt.show()   

MSE2= 1/629 * (np.linalg.norm(np.matmul(X2, beta2) - Y)) ** 2

MSE10 = 0.009348751458203415

pred10 = 0.5087757565416515  

MSE100 = 0.0014547562364331283

pred100 = 0.5089902472113171

MSE1000 = 2.2450937770053657e-28

pred1000 = -7.088223356300462

