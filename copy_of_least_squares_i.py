# -*- coding: utf-8 -*-
"""Copy of Least_squares_I.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PzeIVomLN3TighAA6z_XgRQlySvSWFvp

#**Lab 6 - Least squares I**

Enter your code in the spaces provided. Do not change any of the variable names or function names that are already provided for you. In places where we specify the name of the return value, make sure that your code produces the a value with the correct name.
"""

# Do not edit this cell.

LabID="Lab6"

try:
  from graderHelp import ISGRADEPLOT
except ImportError:
  ISGRADEPLOT = True

"""**Enter your name, section number, and BYU NetID**"""

# Enter your first and last names in between the quotation marks.

first_name="Drew"

last_name="Wilson"

# Enter your Math 215 section number in between the quotation marks. 

section_number="Your Math 215 section number goes here"  

# Enter your BYU NetID in between the quotation marks.  NOT YOUR BYU ID NUMBER! 

BYUNetID="aw053102"

"""**Import NumPy**"""

import numpy as np

"""**Problem 1**"""

# Replace the values of 0 with the NumPy arrays from Problem 1.

X1= np.array([[1,5],[1,10],[1,15],[1,20],[1,25],[1,30],[1,35],[1,40],[1,45],[1,50]])

Y1= np.array([3.33,4.43,4.39,5.23,5.67,6.06,7.01,7.16,8.03,8.78])

"""**Problem 2**"""

# Replace the values of 0 with the normal equation coefficient matrix and normal equation right-hand side respectively from Problem 2.

normal_coef1= np.matmul(np.transpose(X1), X1)

normal_vect1= np.matmul(np.transpose(X1), Y1)

"""**Problem 3**"""

# Replace the value of 0 with the least squares solution beta1 you found in Problem 3.

beta1= np.linalg.solve(normal_coef1, normal_vect1)

# Define a function whose graph is the line of best fit.

def ls1_line(x): 

  return beta1[1]*x + beta1[0]

import matplotlib.pyplot as plt

# Construct your plot of ls1_line and the corresponding data points here. Put all of your code to create the plots inside the function below.

def create_plots1():
  x = np.array(range(50))
  y = ls1_line(x)
  # Put all of your code to create the plots here.
  plt.plot(x, y)
  x2 = np.array(range(5, 55, 5))
  plt.plot(x2, Y1,'o')
  plt.show()
  return None

# Replace the value of 0 with your prediction of the satellite's velocity at t=60.

pred1= ls1_line(60)
print(pred1)
create_plots1()

"""**Problem 4**"""

# Replace the values of 0 with the NumPy arrays from Problem 4.

X2= np.array([[5,5**2],[10,10**2],[15,15**2],[20,20**2],[25,25**2],[30,30**2],[35,35**2],[40,40**2],[45,45**2],[50,50**2]])

Y2= np.array([20.57,87.48,197.45,347.67,546.12,784.35,1066.02,1390.97,1761.85,2177.34])

"""**Problem 5**"""

# Replace the values of 0 with the normal equation coefficient matrix and normal equation right-hand side, and least squares solution from Problem 5.

normal_coef2= np.matmul(np.transpose(X2), X2) 

normal_vect2= np.matmul(np.transpose(X2), Y2)

print(normal_coef2.shape, normal_vect2.shape)

beta2= np.linalg.solve(normal_coef2, normal_vect2)

beta2

"""**Problem 6**"""

# Define a function whose graph is the parabola of best fit.

def ls2_par(x): 
  
  return (beta2[0]*x) + ((beta2[1]) * (x**2))

# Construct your plot of ls2_par and the corresponding data points here. Put all of your code to create the plots inside the function below.

def create_plots2():
  x = np.array(range(50))
  y = ls2_par(x)
  # Put all of your code to create the plots here.
  plt.plot(x, y)
  x2 = np.array(range(5,55,5))
  plt.plot(x2, Y2, 'o')
  plt.show()
  return None 

create_plots2()

# Replace the value of 0 with your prediction of the satellite's position at t=60.

pred2= ls2_par(60)
pred2

"""**Problem 7**"""

# Replace the values of 0 with the NumPy arrays from Problem 7.

X3= np.array([[1,np.log(87.77)],[1,np.log(224.7)],[1,np.log(365.25)],[1,np.log(686.95)],[1,np.log(4332.62)],[1,np.log(10759.2)]])

Y3= np.array([np.log(.389),np.log(.724),np.log(1),np.log(1.524),np.log(5.2),np.log(9.51)])

# Replace the values of 0 with the normal equation coefficient matrix and normal equation right-hand side from Problem 7.

normal_coef3= np.matmul(np.transpose(X3), X3) 

normal_vect3= np.matmul(np.transpose(X3), Y3)

# Replace the value of 0 with the least squares solution from Problem 7.

beta3= np.linalg.solve(normal_coef3, normal_vect3)

"""**Problem 8**"""

# Replace the values of 0 with your predictions for the semi-major axes of Uranus and Neptune.

pred_Uran= np.exp(beta3[0])*30687.15**beta3[1]

pred_Nept= np.exp(beta3[0])*60190.03**beta3[1]

"""**STOP!  BEFORE YOU SUBMIT THIS LAB:**  Go to the "Runtime" menu at the top of this page, and select "Restart and run all".  If any of the cells produce error messages, you will either need to fix the error(s) or delete the code that is causing the error(s).  Then use "Restart and run all" again to see if there are any new errors.  Repeat this until no new error messages show up.

**You are not ready to submit until you are able to select "Restart and run all" without any new error messages showing up.  Your code will not be able to be graded if there are any error messages.**

To submit your lab for grading you must first download it to your compute as .py file. In the "File" menu select "Download .py". The resulting file can then be uploaded to http://www.math.byu.edu:30000 for grading.
"""
