import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import sin

def help():
    print("To use this program, please type the following in your terminal:")
    print("     $ python [the script filename].py [data filename].csv [--help / --h / --plot / --p / --detail / --d / --s]")
    print("\n")
    print("Where [--help / --h / --plot / --p / --detail / --d / --s] are optional terminal parameters:")
    print("\n")
    print("     --plot / --p : Plots the data in line segments of 20 datapoints in different colours")
    print("     --help / --h : Displays this message for guidance on terminal options")
    print("     --detail / --d / --s : Plots the data in line segments of 20 datapoints in different colours and displays an array of the error values at each line segment")
    print("\n")

if(len(sys.argv) < 2 ) : 
    help()

if(len(sys.argv) > 2 ) :         
    if(sys.argv[2] == "--help" or sys.argv[2] == "--h"):
        help()
    if(sys.argv[2] == "--detail" or sys.argv[2] == "--d" or sys.argv[2] == "--s"):
        detail = True
    else :
        detail = False


"""Utility Functions------------------"""
def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values
    """return (xs, ys)"""

def view_data_segments(xs, ys):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()



"""Least Square Functions-------------"""
def Wh(X, Y, XT):
    return np.dot(np.dot(np.linalg.inv(np.dot(XT, X)), XT), Y)

def leastSquaresReg(X, Y, order, pol):
    X = X.reshape((len(X), 1))
    Xcalc = np.ones((len(X), order+1))
    Xcalc[:,1] = X[:, 0]
    XT = np.transpose(Xcalc)

    if pol:
        for j in range(1, order):
            Xcopy = X.copy()
            for i in range(0, len(X)):
                Xcopy[i] = X.item(i)**(j+1)
            Xcalc[:,j+1] = Xcopy[:, 0]
    else:
        Xcopy = X.copy()
        for i in range(0, len(X)):
            Xcopy[i] = sin(X.item(i))
        Xcalc[:,order] = Xcopy[:, 0]
    return Wh(Xcalc, Y, XT)



"""Error Function--------------------"""
def sumSquaredError(X, Y, A, order, pol):
    s = 0
    if pol:
        for i in range(0, len(X) - 1):
            if order==1:
                Yb = A[0] + A[1] * X.item(i)
            if order==2:
                Yb = A[0] + A[1] * X.item(i) + A[2] * X.item(i)**order
            if order==3:
                Yb = A[0] + A[1] * X.item(i) + A[2] * X.item(i)**(order-1) + A[3] * X.item(i)**order
            if order==4:
                Yb = A[0] + A[1] * X.item(i) + A[2] * X.item(i)**(order-2) + A[3] * X.item(i)**(order-1) + A[4] * X.item(i)**order
            s = s + (Yb-Y.item(i))**2
    else:
        for i in range(0, len(X) - 1):
            s = s + ((A[0] + A[1] * sin(X.item(i))) - Y.item(i))**2
    return s



"""Main-------------------------------"""
fig, splt = plt.subplots()
X, Y = load_points_from_file(sys.argv[1])
XS = np.split(X, len(X)/20)
YS = np.split(Y, len(Y)/20)
S = 0
SArray = []
for i in range(len(XS)):
    Xmin = XS[i].min()
    Xmax = XS[i].max() 
    ALinear = leastSquaresReg(XS[i], YS[i], 1, True)
    APolQuad = leastSquaresReg(XS[i], YS[i], 2, True)
    APolCub = leastSquaresReg(XS[i], YS[i], 3, True)
    APolQuar = leastSquaresReg(XS[i], YS[i], 4, True)
    ASin = leastSquaresReg(XS[i], YS[i], 1, False) 

    SLinear = sumSquaredError(XS[i], YS[i], ALinear, 1, True)
    SPolQuad = sumSquaredError(XS[i], YS[i], APolQuad, 2, True)
    SPolCub = sumSquaredError(XS[i], YS[i], APolCub, 3, True)
    SPolQuar = sumSquaredError(XS[i], YS[i], APolQuar, 4, True)
    SSin = sumSquaredError(XS[i], YS[i], ASin, 1, False)

    w1 = [1, 1, 1, 1, 1]
    w2 = [1, 2, 4, 8, 8]
    w3 = [1, 1.5, 2, 4, 1]
    w=w2
    Ss = [SLinear*w[0], SPolQuad*w[1], SPolCub*w[2], SPolQuar*w[3], SSin*w[4]]
    argmin = min(Ss)

    if (argmin == SLinear*w[0]):
        if(i > 0):
            Xmin = XS[i-1].max()
        S = S + sumSquaredError(XS[i], YS[i], ALinear, 1, True)
        Ymin = ALinear[0] + ALinear[1]*Xmin
        Ymax = ALinear[0] + ALinear[1]*Xmax
        splt.plot([Xmin,Xmax], [Ymin,Ymax], '-r', lw=1)
        if detail:
            SArray.append(sumSquaredError(XS[i], YS[i], ALinear, 1, True))
    if (argmin == SPolQuad*w[1]):
        if(i > 0):
            Xmin = XS[i-1].max()
        S = S + sumSquaredError(XS[i], YS[i], APolQuad, 2, True)
        XtoPlot = np.linspace(Xmin, Xmax, len(X))
        YtoPlot = APolQuad[0] + XtoPlot*APolQuad[1] + (XtoPlot**2)*APolQuad[2]
        splt.plot(XtoPlot, YtoPlot, '-r', lw=1)
        if detail:
            SArray.append(sumSquaredError(XS[i], YS[i], APolQuad, 2, True))
    if (argmin == SPolCub*w[2]):
        if(i > 0):
            Xmin = XS[i-1].max()
        S = S + sumSquaredError(XS[i], YS[i], APolCub, 3, True)
        XtoPlot = np.linspace(Xmin, Xmax, len(X))
        YtoPlot = APolCub[0] + XtoPlot*APolCub[1] + (XtoPlot**2)*APolCub[2] + (XtoPlot**3)*APolCub[3]
        splt.plot(XtoPlot, YtoPlot, '-r', lw=1)
        if detail:
            SArray.append(sumSquaredError(XS[i], YS[i], APolCub, 3, True))
    if (argmin == SPolQuar*w[3]):
        if(i > 0):
            Xmin = XS[i-1].max()
        S = S + sumSquaredError(XS[i], YS[i], APolQuar, 4, True)
        XtoPlot = np.linspace(Xmin, Xmax, len(X))
        YtoPlot = APolQuar[0] + XtoPlot*APolQuar[1] + (XtoPlot**2)*APolQuar[2] + (XtoPlot**3)*APolQuar[3] + (XtoPlot**4)*APolQuar[4]
        splt.plot(XtoPlot, YtoPlot, '-r', lw=1)
        if detail:
            SArray.append(sumSquaredError(XS[i], YS[i], APolQuar, 4, True))
    if (argmin == SSin*w[4]):
        S = S + sumSquaredError(XS[i], YS[i], ASin, 1, False)
        XtoPlot = np.linspace(Xmin, Xmax, len(X))
        YtoPlot = ASin[0] + np.sin(XtoPlot)*ASin[1]
        splt.plot(XtoPlot, YtoPlot, '-r', lw=1)
        if detail:
            SArray.append(sumSquaredError(XS[i], YS[i], ASin, 1, False))

print("Error: ")
print(S)
print("\n")
if detail:
    print(SArray)
print("\n")

if(len(sys.argv) > 2 ) :         
    if(sys.argv[2] == "--plot" or sys.argv[2] == "--p"):
        view_data_segments(X, Y)
    if(sys.argv[2] == "--help" or sys.argv[2] == "--h"):
        help()
    if(sys.argv[2] == "--detail" or sys.argv[2] == "--d" or sys.argv[2] == "--s"):
        view_data_segments(X, Y)
    else :
        detail = False