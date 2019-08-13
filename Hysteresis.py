### Cusp.py
### MIT LICENSE 2019 Kelly Spendlove
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import root
import matplotlib.animation as animation



def hyst_rhs(t,state):
  #Right hand side of hysteresis ODE, dx/dt = c+x-x^3, dc/dt = 0
  c,y = state
  return np.array([0,c+y-y**3])

def drift_rhs(t,state,eps):
  #Right hand side of Conley's slow dirft, dx/dt = c+x-x^3, dc/dt = eps*c(c-1)
  c,y = state
  return np.array([eps*c*(c-1),c+y-y**3])

def control_rhs(t,state,eps):
  #Right hand side of controlled hysteresis ODE, dx/dt = c+x-x^3, dc/dt = g(t)
  c,y = state
  return np.array([eps*t*(t-1),c+y-y**3])


def plot_solution(rhs, init_cond,t_final):
  X = integrate.solve_ivp(rhs, [0,t_final], init_cond)
  plt.plot(X.t, np.transpose(X.y))
  plt.xlabel('t')
  plt.ylabel('c, y')
  plt.legend(('c', 'y'))

def plot_trajectory(ax,rhs,init_cond,t_final,lw=2):
  if ax==None:
    fig,ax = plt.subplots()
    ax.set_xlabel('c')
    ax.set_ylabel('y')
  X = integrate.solve_ivp(rhs, [0,t_final], init_cond)
  ax.plot(X.y[0],X.y[1],color='black',lw=lw)
  return ax

def streamplot(ax,rhs,c_bounds,y_bounds,n = 100):
  if ax==None:
    fig,ax = plt.subplots()
    ax.set_xlabel('c')
    ax.set_ylabel('y')
  c = np.linspace(c_bounds[0],c_bounds[1], n)
  y = np.linspace(y_bounds[0],y_bounds[1], n)
  cv,yv = np.meshgrid(c,y)
  c_vel = np.empty_like(cv)
  y_vel = np.empty_like(yv)
  for i in range(cv.shape[0]):
    for j in range(cv.shape[1]):
      c_vel[i,j],y_vel[i,j] = rhs(None,np.array([cv[i,j],yv[i,j]]))
  speed = np.sqrt(c_vel**2+y_vel**2)
  lw = 0.5+2.5*speed / speed.max()
  ax.streamplot(cv,yv,c_vel,y_vel,linewidth=lw,arrowsize=1.2,density=1,color='thistle')
  return ax

def plot_nullclines(ax,x_bounds,y_bounds,n=100):
  if ax==None:
    fig,ax=plt.subplots()
    ax.set_xlabel('c')
    ax.set_ylabel('y')
  y = np.linspace(y_bounds[0],y_bounds[1], n)
  nc_c = -y+y**3
  ax.plot(nc_c,y,lw=2,color='wheat')
  ax.axis([x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]])
  return ax

def plot_fixedpoints(ax,eps,c_bounds,y_bounds):
  if ax==None:
    fig,ax=plt.subplots()
    ax.axis([c_bounds[0], c_bounds[1], y_bounds[0], y_bounds[1]])
    ax.set_xlabel('c')
    ax.set_ylabel('y')
  fps = fixed_points(eps,c_bounds,y_bounds)
  for fp in fps:
    ax.plot(*fp,'.')
  return ax

def fixed_points(eps,c_bounds,y_bounds,n=6):
  #Search for and return fixed points within bounds
  def F(X):
    c,y=X
    return np.array([eps*c*(c-1),c+y-y**3])
  def DF(X):
    c,y=X
    return np.array([[eps*(2*c-1), 0], 
                     [1, 1-3*y**2]])
  #Search for fixed points
  fp = set()
  x = np.linspace(c_bounds[0],c_bounds[1],n)
  y = np.linspace(y_bounds[0],y_bounds[1],n)    
  xv,yv = np.meshgrid(x,y) 
  for i in range(xv.shape[0]):
    for j in range(xv.shape[1]): 
      sol = root(F,[xv[i,j],yv[i,j]],jac=DF)
      fp.add(tuple(np.around(sol.x,decimals=5)))
  return fp

# def animate_control(rhs,c_bounds,y_bounds,t_bounds):
#   fig,ax = fig.subplots()
#   ax.grid()
#   ax.set(xlim=(c_bounds[0],c_bounds[1]),ylim=(y_bounds[0],y_bounds[1]))
#   c = np.linspace(c_bounds[0],c_bounds[1], n)
#   y = np.linspace(y_bounds[0],y_bounds[1], n)
#   t = np.linspace(t_bounds[0],t_bounds[1], n)
#   cv,yv,tv = np.meshgrid(c,y,t)
#   c_vel = np.empty_like(cv)
#   y_vel = np.empty_like(yv)
#   for i in range(cv.shape[0]):
#     for j in range(cv.shape[1]):
#       c_vel[i,j],y_vel[i,j] = rhs(None,np.array([cv[i,j],yv[i,j]]))
#   color_array = np.sqrt(((dx-n)/2)**2 + ((dy-n)/2)**2)
#   q = ax.quiver(c,y,c_vel,y_vel,color_array)

#   def update(rhs, q, c, y):















