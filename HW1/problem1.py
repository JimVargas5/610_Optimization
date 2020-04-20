# Jim Vargas
# MTH 610 Optimization
# Problem 1

import math as m
import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
pdf=PdfPages('Output1.pdf')

x0=[2,2]
a=[1,-1]
N=100
running_ti=np.linspace(0,1,N+1)
t0=0


for k in [0,1]:
    mu1=m.sqrt(a[k]**2 +1)
    mu2=-1*mu1
    A=mu1 - a[k]
    B=2*mu1
    C=m.log(abs((1+mu2-a[k])/(1+mu1-a[k])))/B -1


    def x(t):
        return A*k1*m.exp(mu1*t) + (mu2-a[k])*k2*m.exp(mu2*t)
    def P(t):
        return (
            (-1*a[k]**2 *m.exp(B *t) + a[k]**2 *m.exp(B) + a[k] *m.exp(B *t) - a[k] *m.exp(B) + mu1**2 *m.exp(B *t) - m.exp(B)* mu1**2 - mu1 *m.exp(B *t) - m.exp(B)* mu1) /
            (-1*a[k] *m.exp(B *t) + a[k] *m.exp(B) - mu1 *m.exp(B *t) - m.exp(B) *mu1 + m.exp(B *t) - m.exp(B))
        )
    P0=P(0)
    k2=(
        (x0[k] - P0*x0[k]*A) /
        (mu1 + mu2 - 2*a[k])
    )
    k1=k2 - P0*x0[k]
    def u(t):
        return -1*Lambda(t)
    def Lambda(t):
        return P(t)*x(t)

    
    running_xi=[x0[k]]
    running_Lambdai=[P0*x0[k]]
    running_ui=[-1*P0*x0[k]]
    for ti in running_ti:
        if ti!=0:
            x_next=x(ti)
            Lambda_next=Lambda(ti)
            u_next=u(ti)

            running_xi.append(x_next)
            running_Lambdai.append(Lambda_next)
            running_ui.append(u_next)

    fig=plt.figure()
    plt.plot(running_ti, running_xi, label='x(t_i)')
    plt.plot(running_ti, running_ui, label='u(t_i)')
    plt.plot(running_ti, running_Lambdai, label='lambda(t_i)')
    plt.title("N="+str(N)+"; t_i=0, 1/N, 2/N,..., 1; a="+str(a[k]))
    plt.legend(loc='best')

    plt.close()
    pdf.savefig(fig)

pdf.close()