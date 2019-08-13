### Cusp.py
### MIT LICENSE 2019 Kelly Spendlove
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import root
import matplotlib.animation as animation


class Parameter:
    #ParameterClass
    def __init__(self,A,B,C):
      self.A = A
      self.B = B
      self.C = C

def cusp_rhs(t,state,pi):
  #Right hand side of Cusp Normal Form, dx/dt = A+Bx+Cx^3
  x,l = state
  return np.array([pi.A+pi.B*x+pi.C*x**3])

def plot_solution(pi,init_cond,t_final):
  X = integrate.solve_ivp(lambda t, y:cusp_rhs(t,y,pi), [0,t_final], init_cond)
  plt.plot(X.t, np.transpose(X.y))
  plt.xlabel('t')
  plt.ylabel('x, y')
  plt.legend(('x', 'y'))

# def plot_trajectory(pi,init_cond,t_final):
#   X = integrate.solve_ivp(lambda t, y : cusp_rhs(t,y,pi), [0,t_final], init_cond)
#   plt.plot(X.y[0],X.y[1])
#   plt.xlabel('x')
#   plt.ylabel('y')

def streamplot(pi,x_max,y_max,n = 100):
  x = np.linspace(0,x_max, n)
  y = np.linspace(0,y_max, n)
  xv,yv = np.meshgrid(x,y)
  x_vel = np.empty_like(xv)
  y_vel = np.empty_like(yv)
  for i in range(xv.shape[0]):
    for j in range(xv.shape[1]):
      x_vel[i,j],y_vel[i,j] = cusp_rhs(None,np.array([xv[i,j],yv[i,j]]),pi)
  speed = np.sqrt(x_vel**2+y_vel**2)
  lw = 0.5+2.5*speed / speed.max()
  plt.streamplot(xv,yv,x_vel,y_vel,linewidth=lw,arrowsize=1.2,density=1,color='thistle')
  plt.xlabel('x')
  plt.ylabel('y')

def plot_nullclines(pi,x_max,y_max,n=100):
  x = np.linspace(0,x_max,n)
  y = np.linspace(0,y_max,n)
  nc_x = pi.A1 / (1+y**pi.B)
  nc_y = pi.A2 / (1+x**pi.C)
  plt.plot(x,nc_y,lw=2)
  plt.plot(nc_x,y,lw=2)
  plt.xlabel('x')
  plt.ylabel('y')

def plot_fixedpoints(pi,x_max,y_max):
  fps = fixed_points(pi,x_max,y_max)
  for fp in fps:
    plt.plot(*fp,'.')
  plt.axis([0, x_max, 0, y_max])
  plt.xlabel('x')
  plt.ylabel('y')

def fixed_points(pi,x_max,y_max,n=6):
  #Search for and return fixed points within bounds
  def F(X):
    x,y=X
    return np.array([pi.A1/(1+y**pi.B)-x,pi.A2/(1+x**pi.C)-y])
  def DF(X):
    x,y=X
    return np.array([[-1, -pi.A1*pi.B*y**(pi.B-1)/(1+y**pi.B)**2],
                     [-pi.A2*pi.C*x**(pi.C-1)/(1+x**pi.C)**2, -1]])
  #Search for fixed points
  fp = set()
  x = np.linspace(0,x_max,n)
  y = np.linspace(0,y_max,n)    
  xv,yv = np.meshgrid(x,y) 
  for i in range(xv.shape[0]):
    for j in range(xv.shape[1]): 
      sol = root(F,[xv[i,j],yv[i,j]],jac=DF)
      fp.add(tuple(np.around(sol.x,decimals=5)))
  return fp

#DRY SMELL
def plot_fp(A1,A2,B,C):
  x_max,y_max = 6,6
  pi = Parameter(A1,A2,B,C)
  fps = fixed_points(pi,x_max,y_max)
  for fp in fps:
    plt.plot(*fp,'.')
  plt.axis([0, x_max, 0, y_max])
  plt.xlabel('x')
  plt.ylabel('y')



