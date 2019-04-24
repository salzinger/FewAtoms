import numpy as np
import matplotlib.pyplot as plt

from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check


def fourbasis():

    return np.array([basis(4, 0), basis(4, 1), basis(4, 2), basis(4, 3)], dtype=object)


def pairbasis(m,j):
    
    return tensor(fourbasis()[m],fourbasis()[j])


def fourops():

    one, two, three, four = fourbasis()

    sig11 = one * one.dag()
    sig22 = two * two.dag()
    sig33 = three * three.dag()
    sig44 = four * four.dag()
    sig12 = one * two.dag()
    sig32 = three * two.dag()
    sig43 = four * three.dag()

    return np.array([sig11, sig22, sig33, sig44, sig12, sig32, sig43], dtype=object)


def exchangeop(m,j):

    sigmj = pairbasis(m,j) * pairbasis(j,m).dag()

    return sigmj


def singleop(i,m):
    q=np.array([])
    if i==1:
        q=tensor(identity(4),fourops()[m])
    else:
        q=tensor(fourops()[m],identity(4))
    return q



H_atom1=0.1*(singleop(0,4)+singleop(0,4).dag())+1*(singleop(0,5)+singleop(0,5).dag())+0*(singleop(0,6)+singleop(0,6).dag())
H_atom2=0.1*(singleop(1,4)+singleop(1,4).dag())+1*(singleop(1,5)+singleop(1,5).dag())+0*(singleop(1,6)+singleop(1,6).dag())


H=H_atom1+H_atom2+1*(exchangeop(2,3)+exchangeop(3,2))

print(exchangeop(3,2))


times = np.linspace(0.0, 400, 4000)



result = mesolve(H, pairbasis(3,0), times,[0.3*singleop(0,4),0.3*singleop(1,4)], [singleop(0,0),singleop(0,1),singleop(0,2),singleop(0,3),singleop(1,0),singleop(1,1),singleop(1,2),singleop(1,3)])


fig, ax = plt.subplots()
ax.plot(times, result.expect[0],label="a1sig11");
ax.plot(times, result.expect[1],label="a1sig22");
ax.plot(times, result.expect[2],label="a1sig33");
ax.plot(times, result.expect[3],label="a1sig44");
ax.set_xlabel('Time');
ax.set_ylabel('Expectation value');
leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)

fig1, ax1 = plt.subplots()
ax1.plot(times, result.expect[4],label="a2sig11");
ax1.plot(times, result.expect[5],label="a2sig22");
ax1.plot(times, result.expect[6],label="a2sig33");
ax1.plot(times, result.expect[7],label="a2sig44");
ax1.set_xlabel('Time');
ax1.set_ylabel('Expectation value');
leg1 = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
leg1.get_frame().set_alpha(0.5)

plt.show(fig1)
