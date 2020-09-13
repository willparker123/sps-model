import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import sin

"""Utility funtions
---------------------------------------------------------------------------------"""
def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None, engine='python')
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


"""Least square funtions
---------------------------------------------------------------------------------"""
def least_squares_linear(X, Y) :
     X = X.reshape((len(X), 1))
     x_1 = np.ones((len(X), 2))
     x_1[:,1] = X[:, 0]
     XT = np.transpose(x_1)
     A = np.dot(np.dot(np.linalg.inv(np.dot(XT, x_1)), XT), Y)
     return A


def least_squares_pol(X, Y) :
     X = X.reshape((len(X), 1))
     x_1 = np.ones((len(X), 3))
     x_1[:,1] = X[:, 0]
     
     XC = X.copy()
     for i in range(0, len(X)):
        XC[i] = XC.item(i)**2
     x_1[:,2] = XC[:, 0]
     
     XT = np.transpose(x_1)
     A = np.dot(np.dot(np.linalg.inv(np.dot(XT, x_1)), XT), Y)
     return A
 

def least_squares_sine(X, Y) :
     X = X.reshape((len(X), 1))
     x_1 = np.ones((len(X), 2))
     
     XC = X.copy()
     for i in range(0, len(X)):
        XC[i] = sin(XC.item(i))
     x_1[:,1] = XC[:, 0]
     
     XT = np.transpose(x_1)
     A = np.dot(np.dot(np.linalg.inv(np.dot(XT, x_1)), XT), Y)
     return A


"""Calculate error funtions
---------------------------------------------------------------------------------"""
def calculate_error_sine(X, Y, A):
    s = 0
    for i in range(0, len(X) - 1):         
         s = s + ((A[0] + A[1] * sin(X.item(i))) - Y.item(i))**2
    return s
    
    
def calculate_error_linear(X, Y, A):
    s = 0
    for i in range(0, len(X) - 1):         
         s = s + ((A[0] + A[1] * X.item(i)) - Y.item(i))**2
    return s

    
def calculate_error_pol(X, Y, A):
    s = 0
    for i in range(0, len(X) - 1):         
         s = s + ((A[0] + A[1] * X.item(i) + A[2] * X.item(i) ** 2) - Y.item(i))**2
    return s


"""MAIN FUNTION
---------------------------------------------------------------------------------"""
X, Y = load_points_from_file(sys.argv[1])
XS = np.split(X, len(X)/20)
YS = np.split(Y, len(Y)/20)
fig, ax = plt.subplots()
ax.scatter(X, Y, s=200)
S = 0
for i in range(len(XS)):
        Xmin = XS[i].min()
        Xmax = XS[i].max()
        
        A_linear = least_squares_linear(XS[i],YS[i])
        A_pol = least_squares_pol(XS[i],YS[i])
        A_sine = least_squares_sine(XS[i],YS[i])
        
        S_linear = calculate_error_linear(XS[i], YS[i], A_linear)
        S_pol = calculate_error_pol(XS[i], YS[i], A_pol)
        S_sine = calculate_error_sine(XS[i], YS[i], A_sine)
       
        if(S_linear < S_pol*1.5 and S_linear < S_sine):
          if(i > 0):
              Xmin = XS[i-1].max()
          S = S + (calculate_error_linear(XS[i], YS[i], A_linear))
          Ymin = A_linear[0] + A_linear[1]*Xmin
          Ymax = A_linear[0] + A_linear[1]*Xmax
          ax.plot([Xmin,Xmax], [Ymin,Ymax], '-r', lw=4)
        
        else:
          if(S_pol < S_sine):
            if(i > 0):
              Xmin = XS[i-1].max()
            
            S = S + calculate_error_pol(XS[i], YS[i], A_pol)
            Xnew = np.linspace(Xmin, Xmax, len(X))
            Ynew = A_pol[0] + Xnew*A_pol[1] + (Xnew**2)*A_pol[2]
            ax.plot(Xnew, Ynew, '-r', lw=4)
          
          else:
            S = S + calculate_error_sine(XS[i], YS[i], A_sine)
            Xnew = np.linspace(Xmin, Xmax, len(X))
            Ynew = A_sine[0] + np.sin(Xnew)*A_sine[1]
            ax.plot(Xnew, Ynew, '-r', lw=4)

if(len(sys.argv) > 2 ) :         
   if(sys.argv[2] == "--plot"):
      plt.show()
print(S)
print("\n")
