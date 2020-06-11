# Jim Vargas
# MTH 610
# Final Project



import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
pdf=PdfPages('raw_graphs_n_such.pdf')



"""
    This first part solves the Lorenz '63 model using a(/the?) Runge-Kutta-4 method.
        X(t)=[x(t),y(t),z(t)]^T
    Note that X'(t)=f(X,t,p), where p is a vector consiting of the parameters.
    This solution will be treated as the "true" solution later, so everything is
    called "true."
"""

t0=0
tf=100
N_true=10000
h_bar=0.01
t_true=np.linspace(t0, tf, N_true+1) # not necessary?

P=np.array([[10,28,8/3]]) # [sigma, rho, beta]
P=P.T 

X0_true=np.array([[1,1,1]]) # [x, y, z]
X0_true=X0_true.T 

X_true=X0_true
X_true_storage=np.zeros((3,N_true+1))
X_true_storage[:,0]=X0_true.reshape(3)

def f(X_i, p):
    x=X_i[0]
    y=X_i[1]
    z=X_i[2]

    sigma=p[0]
    rho=p[1]
    beta=p[2]

    x_new=sigma*(y - x)
    y_new=rho*x - y - x*z
    z_new=x*y - beta*z
    X_new=np.array([x_new, y_new, z_new])
    return X_new

def RK4(X_i, p, h):
    K1=h*f(X_i, p)
    K2=h*f(X_i + K1/2, p)
    K3=h*f(X_i + K2/2, p)
    K4=h*f(X_i + K3, p)
    return X_i + (K1 + 2*K2 + 2*K3 + K4)/6



for i in range(1, N_true+1):
    X_true=RK4(X_true, P, h_bar)
    X_true_storage[:,i]=X_true.reshape(3)



fig_true=plt.figure()
ax=plt.axes(projection="3d")
ax.scatter3D(
    X_true_storage[0,:], X_true_storage[1,:], X_true_storage[2,:],
    s=1, marker="."
)
ax.set_xlabel('x(t_i)')
ax.set_ylabel('y(t_i)')
ax.set_zlabel('z(t_i)')
plt.title(
    "Solution for X(t)=[x,y,z] using RK4\n"+
    "t_i="+str(t0)+":"+str(tf)+", N="+str(N_true)
)
#plt.show()
#plt.close()
#pdf.savefig(fig_true)



"""
    This next part solves the same problem using an extended Kalman filter, using
    the previous "true" solution to sample as data. These variables here are called
    "model."

    I will do three runs at once, where I observe only the first component of the
    true data, then the first and second components, then all three. These will be
    distinguished with names "one," "two," "three."

    I will also simultaneously do a 'just forecast' run, where I just use the RK4
    model again but with less data points.
"""

h_model=0.1
N=1000

std_dev_b=1
std_dev_obs=0.1
std_dev_model=0.1

Q=(std_dev_model**2)*np.identity(3)

DATA=X_true_storage[:,0::10]

H1=np.array([
    [1,0,0]
])
H2=np.array([
    [1,0,0],
    [0,1,0]
])
H3=np.identity(3)

R1=std_dev_obs**2
R2=(std_dev_obs**2)*np.ones(2)
R3=(std_dev_obs**2)*np.ones(3)



# knock three birds with one for loop
# observe the first component, first and second, first and second and third
X0=X0_true + np.random.normal(0, std_dev_b, size=(3,1))

Xa_one=X0
Xa_two=X0
Xa_three=X0

Xf_just_forecast=X0

Xa_one_storage=np.zeros((3,N+1))
Xa_one_storage[:,0]=X0.reshape(3)
Xa_two_storage=np.zeros((3,N+1))
Xa_two_storage[:,0]=X0.reshape(3)
Xa_three_storage=np.zeros((3,N+1))
Xa_three_storage[:,0]=X0.reshape(3)

Xf_storage=np.zeros((3,N+1))
Xf_storage[:,0]=X0.reshape(3)

Pa=np.identity(3)
Pa_one=Pa
Pa_two=Pa
Pa_three=Pa

def compute_Xf_next(Xa_i):
    return RK4(Xa_i, P, h_model)

def dfdX_i(X_i, p): # 3x3 matrix, X derivative of f(X)
    x=np.asscalar(X_i[0])
    y=np.asscalar(X_i[1])
    z=np.asscalar(X_i[2])

    sigma=np.asscalar(p[0])
    rho=np.asscalar(p[1])
    beta=np.asscalar(p[2])

    return np.array([
        [-1*sigma, sigma, 0],
        [rho-z, -1, -1*x],
        [y, x, -1*beta]
    ])

def compute_M_i(X_i, p, h): # 3x3 matrix, X derivative of the RK4 thing
    C=h*dfdX_i(X_i, p)
    #print(C)
    return np.identity(3) + C + (C@C)/2 + (C@C@C)/6 + (C@C@C@C)/24

def compute_Pf_i(Pa_i, X_i, p, h):
    M_i=compute_M_i(X_i, p, h)
    return M_i @ Pa_i @ M_i.T + Q

def compute_K_i(Pf_i, H, R):
    return Pf_i @H.T @ np.linalg.inv(H@Pf_i@H.T + R)

def compute_Xa_next(Xf_next, K, H, y):
    return Xf_next + K@(y - H@Xf_next)

def compute_Pa_i(H, Pf_i, R, K_i):
    return (np.identity(3) - K_i @ H) @ Pf_i





for i in range(1, N+1):
    randomV=np.random.normal(0,std_dev_obs, size=(3,1))

    # just the forecast, doesn't matter what H is
    Xf_just_forecast=compute_Xf_next(Xf_just_forecast)

    Xf_storage[:,i-1]=Xf_just_forecast.reshape(3)

    # observe first component only
    Xf_one=compute_Xf_next(Xa_one)
    Pf_one=compute_Pf_i(Pa_one, Xa_one, P, h_model)
    y_one=H1@DATA[:,i].reshape(3,1) + randomV[0]
    K_one=compute_K_i(Pf_one, H1, R1)
    Xa_one=compute_Xa_next(Xf_one, K_one, H1, y_one)
    Pa_one=compute_Pa_i(H1, Pf_one, R1, K_one)

    Xa_one_storage[:,i]=Xa_one.reshape(3)

    # observe first and second
    Xf_two=compute_Xf_next(Xa_two)
    Pf_two=compute_Pf_i(Pa_two, Xa_two, P, h_model)
    y_two=H2@DATA[:,i].reshape(3,1) + randomV[0:1]
    K_two=compute_K_i(Pf_two, H2, R2)
    Xa_two=compute_Xa_next(Xf_two, K_two, H2, y_two)
    Pa_two=compute_Pa_i(H2, Pf_two, R2, K_two)

    Xa_two_storage[:,i]=Xa_two.reshape(3)

    # observe all components
    Xf_three=compute_Xf_next(Xa_three)
    Pf_three=compute_Pf_i(Pa_three, Xa_three, P, h_model)
    y_three=H3@DATA[:,i].reshape(3,1) + randomV
    K_three=compute_K_i(Pf_three, H3, R3)
    Xa_three=compute_Xa_next(Xf_three, K_three, H3, y_three)
    Pa_three=compute_Pa_i(H3, Pf_three, R3, K_three)

    Xa_three_storage[:,i]=Xa_three.reshape(3)
    


# just the forcast
'''fig_model_forecast_only=plt.figure()
ax=plt.axes(projection="3d")
ax.scatter3D(
    Xf_storage[0,:], Xf_storage[1,:], Xf_storage[2,:],
    s=1, marker="."
)
ax.set_xlabel('x(t_i)')
ax.set_ylabel('y(t_i)')
ax.set_zlabel('z(t_i)')
plt.title(
    "Solution for X(t)=[x,y,z]\n"+
    "just the forecasted values\n"+
    "t_i="+str(t0)+":"+str(tf)+", N="+str(N)
)'''

# first component only
fig_model_one=plt.figure()
ax=plt.axes(projection="3d")
ax.scatter3D(
    Xa_one_storage[0,:], Xa_one_storage[1,:], Xa_one_storage[2,:],
    s=1, marker="."
)
ax.set_xlabel('x(t_i)')
ax.set_ylabel('y(t_i)')
ax.set_zlabel('z(t_i)')
plt.title(
    "Solution for X(t)=[x,y,z] using EKF\n"+
    "H observes just first component\n"+
    "t_i="+str(t0)+":"+str(tf)+", N="+str(N)
)

# first and second components
fig_model_two=plt.figure()
ax=plt.axes(projection="3d")
ax.scatter3D(
    Xa_two_storage[0,:], Xa_two_storage[1,:], Xa_two_storage[2,:],
    s=1, marker="."
)
ax.set_xlabel('x(t_i)')
ax.set_ylabel('y(t_i)')
ax.set_zlabel('z(t_i)')
plt.title(
    "Solution for X(t)=[x,y,z] using EKF\n"+
    "H observes just first and second components\n"+
    "t_i="+str(t0)+":"+str(tf)+", N="+str(N)
)

# observe all components
fig_model_three=plt.figure()
ax=plt.axes(projection="3d")
ax.scatter3D(
    Xa_three_storage[0,:], Xa_three_storage[1,:], Xa_three_storage[2,:],
    s=1, marker="."
)
ax.set_xlabel('x(t_i)')
ax.set_ylabel('y(t_i)')
ax.set_zlabel('z(t_i)')
plt.title(
    "Solution for X(t)=[x,y,z] using EKF\n"+
    "H observes all components\n"+
    "t_i="+str(t0)+":"+str(tf)+", N="+str(N)
)



'''
    Now it's time for some error analysis plots and stuff
'''

# Xf-X in all three
fig_Xf_minus_X, ax=plt.subplots(1,3, sharey=True)
fig_Xf_minus_X.suptitle("Xf-X \n in x, y, z")

difference_x=Xf_storage[0,:]-DATA[0,:]
difference_y=Xf_storage[1,:]-DATA[1,:]
difference_z=Xf_storage[2,:]-DATA[2,:]
running_average_x=[]
running_average_y=[]
running_average_z=[]
for k in range(0,901):
    running_average_x.append(np.mean(difference_x[k:k+100]))
    running_average_y.append(np.mean(difference_x[k:k+100]))
    running_average_z.append(np.mean(difference_x[k:k+100]))

t_model=np.linspace(0,900,901)
ax[0].plot(t_model, running_average_x)
ax[1].plot(t_model, running_average_y)
ax[2].plot(t_model, running_average_z)

# Xa-X in x
fig_Xa_minus_X_x, ax=plt.subplots(1,3, sharey=True)
fig_Xa_minus_X_x.suptitle("Xa-X in x\n one, two, three")

difference_one=Xa_one_storage[0,:]-DATA[0,:]
difference_two=Xa_two_storage[0,:]-DATA[0,:]
difference_three=Xa_three_storage[0,:]-DATA[0,:]
running_average_one=[]
running_average_two=[]
running_average_three=[]
for k in range(0,901):
    running_average_one.append(np.mean(difference_one[k:k+100]))
    running_average_two.append(np.mean(difference_two[k:k+100]))
    running_average_three.append(np.mean(difference_three[k:k+100]))

t_model=np.linspace(0,900,901)
ax[0].plot(t_model, running_average_one)
ax[1].plot(t_model, running_average_two)
ax[2].plot(t_model, running_average_three)

# Xa-X in y
fig_Xa_minus_X_y, ax=plt.subplots(1,3, sharey=True)
fig_Xa_minus_X_y.suptitle("Xa-X in y\n one, two, three")

difference_one=Xa_one_storage[1,:]-DATA[1,:]
difference_two=Xa_two_storage[1,:]-DATA[1,:]
difference_three=Xa_three_storage[1,:]-DATA[1,:]
running_average_one=[]
running_average_two=[]
running_average_three=[]
for k in range(0,901):
    running_average_one.append(np.mean(difference_one[k:k+100]))
    running_average_two.append(np.mean(difference_two[k:k+100]))
    running_average_three.append(np.mean(difference_three[k:k+100]))

t_model=np.linspace(0,900,901)
ax[0].plot(t_model, running_average_one)
ax[1].plot(t_model, running_average_two)
ax[2].plot(t_model, running_average_three)

# Xa-X in z
fig_Xa_minus_X_z, ax=plt.subplots(1,3, sharey=True)
fig_Xa_minus_X_z.suptitle("Xa-X in z\n one, two, three")

difference_one=Xa_one_storage[2,:]-DATA[2,:]
difference_two=Xa_two_storage[2,:]-DATA[2,:]
difference_three=Xa_three_storage[2,:]-DATA[2,:]
running_average_one=[]
running_average_two=[]
running_average_three=[]
for k in range(0,901):
    running_average_one.append(np.mean(difference_one[k:k+100]))
    running_average_two.append(np.mean(difference_two[k:k+100]))
    running_average_three.append(np.mean(difference_three[k:k+100]))

t_model=np.linspace(0,900,901)
ax[0].plot(t_model, running_average_one)
ax[1].plot(t_model, running_average_two)
ax[2].plot(t_model, running_average_three)





plt.show()
print("The end")