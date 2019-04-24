import numpy as np
import matplotlib.pyplot as plt

from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check

def productstate(j,N):
  basislist=[]
  for n in range(0,N-1):
    basislist.append(basis(2,1))
  basislist.insert(j,basis(2,0))
  return tensor(basislist).unit()

def bellstate(i,j,N):  
  blist1=[]
  blist2=[]
  for n in range(0,N-2):
    blist1.append(identity(2))
    blist2.append(identity(2))

  blist1.insert(i,basis(2,0))
  blist1.insert(j,basis(2,1))

  blist2.insert(i,basis(2,1))
  blist2.insert(j,basis(2,0))

  bell=tensor(blist1)+tensor(blist2)

  return bell.unit()

def askatom(j,N):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(2))
  oplist.insert(j,sigmaz())
  return tensor(oplist)

def disorder(E,N):
  oplist=[]
  for n in range(0,N):
    oplist.append(E*np.random.random([1])[0]*basis(2,0)*basis(2,0).dag()+E*np.random.random([1])[0]*basis(2,1)*basis(2,1).dag())
  return tensor(oplist)

def up(j,N):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(2))
  oplist.insert(j,sigmap())
  return tensor(oplist)

def down(j,N):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(2))
  oplist.insert(j,sigmam())
  return tensor(oplist)

def V(R,a,j,k):
  return (R/float((a*np.abs(k-j))))**3

def J_eff(R,a,j,k):
  return 1*V(R,a,j,k)/(2+2*V(R,a,j,k)**2)

def G_eff(R,a,j,k):
  return 0*V(R,a,j,k)**2/(1+V(R,a,j,k)**2)**2

def g_eff(R,a,j,k):
  return 1*V(R,a,j,k)**4/(1+V(R,a,j,k)**2)**2

def H_eff(R,a,N):
  H=0
  h=0
  for k in range(0,N):
    for j in range(0,N):
      if k>j:        
        h+1
        H=H+J_eff(R,a,j,k)*(up(j,N)*down(k,N)+up(k,N)*down(j,N))

        #print("H",h,": Coherent exchange dynamics between sites k=",k,"j=",j,"at distance",a*np.abs(k-j),"with J_eff:","%6.5f" % J_eff(R,a,j,k),H)

  return H

scatterindex=[]

def L_eff(R,a,N):
  L=[]
  nlist=0
  sitelist=[]
  for j in range(0,N):
    for k in range(0,N):
      if k>j:

        L.append(np.sqrt(G_eff(R,a,j,k))*(up(k,N)*down(j,N)))

        scatterindex.append(k)

        #print("L",len(L)-1,": Transfer to scattering site k=",k,"from previous site j=",j," at distance=",a*np.abs(k-j),"with G_eff:","%6.5f" % G_eff(R,a,j,k))

        L.append(np.sqrt(G_eff(R,a,j,k))*(up(j,N)*down(k,N)))

        scatterindex.append(j)

        #print("L",len(L)-1,": Transfer to scattering site j=",j,"from previous site k=",k,"at distance=",a*np.abs(k-j),"with G_eff:","%6.5f" % G_eff(R,a,j,k))
       
      if k!=j:

        L.append(-1j*np.sqrt(g_eff(R,a,j,k))*up(k,N)*down(k,N))

        scatterindex.append(j)

        #print("L",len(L)-1,": Scattering event at j=",j,"due to impurity site k=",k,"at distance=",a*np.abs(k-j),"with g_eff:","%6.5f" % g_eff(R,a,j,k))


  return L


N=8
a=1
timesteps=200
ntraj=200
statelist=[]
opts=Options(store_states=True,store_final_state=True,ntraj=200)



tracedEEav=np.zeros(timesteps)
tracedGGav=np.zeros(timesteps)
purityAav=np.zeros(timesteps)
purityBav=np.zeros(timesteps)
purityCav=np.zeros(timesteps)
concav=np.zeros(timesteps)
VNav=np.zeros(timesteps)
disavgs=1
i=1

for r in [1.0]:
  for t in np.ones(disavgs):
    print(i,"of",disavgs)
    i=i+1
    #print("Interaction Coefficient over one-half EIT bandwidth: ","%6.3f" % (r**(1/3)))
    #print("Interparticle spacing in units of [Critical Radius R]: ", "%6.3f" % (a/r))
    times = np.linspace(0.0, t*100, timesteps)
    results=[]
    asklist=[]

    for j in range(0,N):
      asklist.append(askatom(j,N))

    #result = mcsolve(H_eff(r,a,N) , productstate(0,N) , times, L_eff(r,a,N) , asklist, options=opts)

    result = mesolve(H_eff(r,a,N) , productstate(0,N) , times, L_eff(r,a,N), [], options=opts)

    #print(ket2dm(productstate(0,N)))
    #print(ket2dm(productstate(1,N)))
   
    tracedEE=[]
    tracedGG=[]
    purityA=[]
    purityB=[]
    purityC=[]
    conc=[]
    VN=[]

    #summed=0
    #for nt in range(0,ntraj):
      #summed=summed+result.states[nt]

    for t in range(0,timesteps):

      #print(result.states[t].ptrace(0))

      tracedEE.append(np.abs(result.states[t].ptrace(7)[0][0][0]))
      #tracedGG.append(np.abs(result.states[t].ptrace(0)[1][0][1]))
      #conc.append(concurrence(result.states[t]))
      VN.append(entropy_vn(result.states[t]))

      rhoA=result.states[t].ptrace(0)
      rhoAsquare=rhoA*rhoA
      purityA.append(rhoAsquare[0][0][0]+rhoAsquare[1][0][1])

      rhoB=result.states[t].ptrace(1)
      rhoBsquare=rhoB*rhoB
      purityB.append(rhoBsquare[0][0][0]+rhoBsquare[1][0][1])

      rhoC=result.states[t].ptrace(2)
      rhoCsquare=rhoC*rhoC
      purityC.append(rhoCsquare[0][0][0]+rhoCsquare[1][0][1])

    purityAav=np.add(purityAav,purityA)
    purityBav=np.add(purityBav,purityB)
    purityCav=np.add(purityCav,purityC)
    #concav=np.add(concav,conc)
    VNav=np.add(VNav,VN)

    #fig, ax = plt.subplots()

    #ax.plot(times, tracedEE, label="rho e1e1")
    #ax.plot(times, tracedGG, label="rho g1g1")
    #ax.plot(times, np.abs(purityA), label="PurityA")
    #ax.plot(times, np.abs(purityB), label="PurityB")
    #ax.plot(times, conc, label="Concurrence")
    #leg = plt.legend(loc='best', ncol=3, shadow=True, fancybox=True)
    #leg.get_frame().set_alpha(0.5)
    #plt.show()

    #for n in range(0,N):
        #results.append(result.expect[n])
        #ax.plot(times, results[n], label="n=%d"%(n,))

    #leg = plt.legend(loc='best', ncol=3, shadow=True, fancybox=True)
    #leg.get_frame().set_alpha(0.5)

    #ax.set_xlabel('Time');
    #ax.set_ylabel('Expectation value sigma(z)');

    #plt.savefig("evolution at t="+str(t*10)+" R_crit"+str(r)[0]+"."+str(r)[2]+".pdf", dpi=None, facecolor='w', edgecolor='w',
        #orientation='portrait', papertype=None, format=None,
        #transparent=False, bbox_inches=None, pad_inches=0.1,
        #frameon=None, metadata=None)
    #plt.gcf().clear()

    #statelist.append(Qobj(inpt=result.states, dims=[[len(result.states)], [1]], shape=[], type=None, isherm=None, fast=False, superrep=None))

    #scatterlist=[]
    
    #for o in range(0,ntraj):
      #for g in range(0,len(config.operlist[o])):
        #print("operator",config.operlist[o][g],"scattersite",scatterindex[config.operlist[o][g]])
        #scatterlist.append(scatterindex[config.operlist[o][g]])


    #plt.hist(scatterlist, bins=range(0,N+1),rwidth=0.5,align='right',weights=np.ones(len(scatterlist))/ntraj)
    #plt.title("Average scattered photons at time t="+str(t*10)+" with R_crit="+str(r))

    #plt.savefig("histogram at t= "+str(t*10)+" R_crit"+str(r)[0]+str(r)[2]+".pdf", dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None, metadata=None)

    #plt.gcf().clear()
    #fig, ax = plt.subplots()

fig, ax = plt.subplots()
ax.plot(times, np.abs(tracedEE)/disavgs, label="rho_ee_site_8")
#ax.plot(times, np.abs(purityAav)/disavgs, label="Purity_site_1")
#ax.plot(times, np.abs(purityBav)/disavgs, label="PurityB")
#ax.plot(times, np.abs(purityCav)/disavgs, label="PurityC")
#ax.plot(times, concav/disavgs, label="Concurrence")
ax.plot(times, VNav/disavgs, label="Von-Neumann Entropy")
leg = plt.legend(loc='best', ncol=3, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()
