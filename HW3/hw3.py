# Jim Vargas
# MTH 610
# HW 3

# Remember that vectors are actually 2D at least, nx1 or 1xn matrices


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pdf=PdfPages('raw_graphs_n_such.pdf')




''' constants '''
h=1
t0=0
tf=200
N=1000
t=np.linspace(t0, tf, int(tf/h)+1)
Gamma=0.04

stand_dev=0.001
Q=np.diag(np.array([0.01, 0.01, 0.01, 0.04]))

DATA=np.loadtxt('obsdata_txt.txt')




''' data structures and models '''
Xa_0=np.array([[997, 3, 0, 0.1]]) # [S I R Beta]
Xa_0=Xa_0.transpose() # [S I R Beta]^T
Xa=Xa_0
Xa_storage=np.zeros((4, tf+1))
Xa_storage[:,0]=Xa.reshape(4)
Xf_just_forecast=Xa
Xf_storage=np.zeros((4, tf+1))
Xf_storage[:,0]=Xa.reshape(4)

Pa_0=np.diag(np.array([10, 10, 0.01, 0.04]))
Pa=np.array(Pa_0)

def Mf(X): # 4x1
    'input is augmented analysis Xa_k-1'
    'output is forecast Xf_k'
    S=X[0]
    I=X[1]
    R=X[2]
    Beta=X[3]
    c=h/N

    S_next= S - c*Beta*S*I
    I_next= I + c*Beta*S*I - h*Gamma*I
    R_next= R + h*Gamma*I
    Beta_next= Beta
    return np.array([S_next, I_next, R_next, Beta_next]).reshape((4,1))

def Ma(X): # 4x4
    'input is augmented analysis Xa_k-1'
    'output is {dMf/dXa}_k-1, for decluttering code'
    S=X[0]
    I=X[1]
    Beta=X[3]
    c=h/N
    return np.array([
        [1-c*Beta*I, -1*c*Beta*S, 0, -1*c*S*I],
        [c*Beta*I, 1+c*Beta*S-h*Gamma, 0, c*S*I],
        [0, h*Gamma, 1, 0],
        [0, 0, 0, 1]
    ])

def Pf(P, X):
    'input is augmented analysis Pa_k-1, augmented analysis Xa_k-1'
    'output is forecast covariance Pf_k'
    return np.matmul(
        Ma(X), np.matmul( P, Ma(X).transpose() )
    ) + Q

H=np.array([[0,0,1,0]]) # observe R only
r_constant=stand_dev**2

def Xa_k(X, P, K, y):
    'input is forecast Xf_k, forecast covariance Pf_k'
    'output is augmented analysis Xa_k'
    return X + K*(y - np.matmul(H,X))

def Pa_k(P,K):
    'input is forecast covariance Pf_k'
    'output is analysis covariance Pa_k'
    return np.matmul(
        (np.identity(4) - np.matmul(K,H)), P
    )




''' main loop '''
for k in range(1, tf+1):
    Xf_just_forecast=Mf(Xf_just_forecast)
    Xf_k=Mf(Xa)
    Pf_k=Pf(Pa, Xa)

    K=np.matmul( Pf_k, H.transpose() )
    c=np.matmul(
        H, np.matmul( Pf_k, H.transpose() )
    )
    K=((c+r_constant)**(-1))*K

    y=DATA[k-1]
    Xa=Xa_k(Xf_k, Pf_k, K, y)
    Pa=Pa_k(Pf_k, K)

    Xf_storage[:,k]=Xf_just_forecast.reshape(4)
    Xa_storage[:,k]=Xa.reshape(4)




''' plots '''
fig_Beta=plt.figure()
plt.plot(t, Xa_storage[3,:], label='Beta(t_k)')
plt.title("Estimate for Beta(t_k)"+'\n'+"Gamma="+str(Gamma))
plt.xlabel("t_k=0:"+str(tf)+", step size="+str(h))
plt.grid(True)
plt.close()
pdf.savefig(fig_Beta)

fig_state=plt.figure()
plt.plot(t, Xa_storage[0,:], label='S(t_k)')
plt.plot(t, Xa_storage[1,:], label='I(t_k)')
plt.plot(t, Xa_storage[2,:], label='R(t_k)')
plt.title("Estimate for X(t_k) after filter, X=[S,I,R]"+'\n'+"Gamma="+str(Gamma))
plt.xlabel("t_k=0:"+str(tf)+", step size="+str(h))
plt.ylabel("Portion of population N="+str(N))
plt.legend(loc='best')
plt.grid(True)
plt.close()
pdf.savefig(fig_state)

fig_forecasted=plt.figure()
plt.plot(t, Xf_storage[0,:], label='S(t_k)')
plt.plot(t, Xf_storage[1,:], label='I(t_k)')
plt.plot(t, Xf_storage[2,:], label='R(t_k)')
plt.title("Estimate for X(t_k), just the forecast")
plt.xlabel("t_k=0:"+str(tf)+", step size="+str(h))
plt.ylabel("Portion of population N="+str(N))
plt.legend(loc='best')
plt.grid(True)
plt.close()
pdf.savefig(fig_forecasted)

pdf.close()




# My function names are confusing, maybe change them to 'compute_Xa_k' or something...

