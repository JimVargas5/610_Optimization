# Jim Vargas
# MTH 610 Optimization
# Problem 2

import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
pdf=PdfPages('Output2.pdf')

a=np.array([1,-1])
x0=np.array([2,2])
N=100
h=1/N


S_N=1
S=[S_N]
def S_i(x,k):
    if k==1:
        #print("All done")
        return
    else:
        k=k-1
        S_next=h + (1+h*a[0])**2 *x / (1 + h*x)
        S.insert(0, S_next)
        S_i(S_next,k)
S_i(S_N, N)


for k in [0,1]:
    running_x=[x0[k]]
    running_lambda=[]
    running_u=[]
    x=x0[0]
    running_t=np.linspace(0,1,N+1)

    for i in range(0,N):
        x_next=(1 + h*a[k]) * x / (1 + h*S[i])
        running_x.append(x_next)

        Lambda_next=S[i]*x_next
        running_lambda.append(Lambda_next)

        u_i=-1*Lambda_next
        running_u.append(u_i)



    fig=plt.figure()
    plt.plot(running_t, running_x, label='x')
    running_t=np.linspace(h,1,N)
    plt.plot(running_t, running_u, label='u')
    plt.plot(running_t, running_lambda, label='lambda')
    plt.title("N="+str(N)+", a="+str(a[k]))
    plt.legend(loc='best')

    plt.close()
    pdf.savefig(fig)

pdf.close()
