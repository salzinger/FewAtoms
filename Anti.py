import numpy as np
import matplotlib.pyplot as plt

from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check

def productstate(j,N):
  basislist=[]
  for n in range(0,N-1):
    basislist.append(basis(3,0))
  basislist.insert(j,basis(3,1))
  print(basislist)
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
    oplist.append(identity(3))
  oplist.insert(j,Qobj([[0,0,0],[0,-1,0],[0,0,0]]))
  return tensor(oplist)

def disorder(E,N):
  oplist=[]
  for n in range(0,N):
    oplist.append(E*np.random.random([1])[0]*basis(2,0)*basis(2,0).dag()+E*np.random.random([1])[0]*basis(2,1)*basis(2,1).dag())
  return tensor(oplist)

def upXY(j,N,C):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(3))
  oplist.insert(j,Qobj([[0,0,0],[C,0,0],[0,0,0]]))
  return tensor(oplist)

def downXY(j,N,C):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(3))
  oplist.insert(j,Qobj([[0,C,0],[0,0,0],[0,0,0]]))

  return tensor(oplist)

def upJ(j,N,Oc):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(3))
  oplist.insert(j,Qobj([[0,0,0],[0,0,Oc],[0,0,0]]))
  return tensor(oplist)

def downJ(j,N,Oc):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(3))
  oplist.insert(j,Qobj([[0,0,0],[0,0,0],[0,Oc,0]]))
  return tensor(oplist)

def upD(j,N,Oc):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(3))
  oplist.insert(j,Qobj([[0,0,0],[0,0,Oc],[0,0,0]]))
  return tensor(oplist)

def downD(j,N,Oc):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(3))
  oplist.insert(j,Qobj([[0,0,0],[0,0,0],[0,Oc,0]]))
  return tensor(oplist)

def coupling(j,N,Oc):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(3))
  oplist.insert(j,Qobj([[0,0,0],[0,0,Oc],[0,Oc,0]]))
  return tensor(oplist)


def scatter(j,N,Gamma):
  oplist=[]
  for n in range(0,N-1):
    oplist.append(identity(3))
  oplist.insert(j,Qobj([[0,0,0],[0,0,0],[0,0,Gamma]]))
  return tensor(oplist)

def V(R,a,j,k):
  return (R/float((a*np.abs(k-j))))**3

def J_eff(R,a,j,k):
  return 1*V(R,a,j,k)/(2+2*V(R,a,j,k)**2)

def G_eff(R,a,j,k):
  return 1*V(R,a,j,k)**2/(1+V(R,a,j,k)**2)**2

def g_eff(R,a,j,k):
  return 1*V(R,a,j,k)**4/(1+V(R,a,j,k)**2)**2

def H(R,N,C):
  HH=0
  hh=0
  for j in range(0,N-1):        
    hh+1
    HH=HH+0.5*(upXY(j,N,C)*downXY(j+1,N,C)+upXY(j+1,N,C)*downXY(j,N,C))

    #print("H",hh,": Coherent XY dynamics between sites j=",j,"j+1=",j+1)

  return HH

def H_eff(R,a,N):
  H_e=0
  h_e=0
  for k in range(0,N):
    H_e=H_e+coupling(k,N,1)
    for j in range(0,N):
      if k>j:        
        h_e+1
        J=np.sqrt(J_eff(R,a,j,k))
        #H_e=H_e+0.5*(upJ(j,N,J)*downJ(k,N,J)+upJ(k,N,J)*downJ(j,N,J))

        #print("H",h_e,": Exchange dynamics between sites k=",k,"j=",j,"at distance",a*np.abs(k-j),"with J_eff:","%6.5f" % J_eff(R,a,j,k))

  print(H_e)
  return H_e

scatterindex=[]

def L_eff(R,a,N):
  L=[]
  nlist=0
  sitelist=[]

  for j in range(0,N):
    L.append(scatter(j,N,1))
    for k in range(0,N):
      if k>j:

        G=np.sqrt(G_eff(R,a,j,k))

        #L.append(upD(k,N,G)*downD(j,N,G))

        scatterindex.append(k)

        #print("L",len(L)-1,": Transfer to scattering site k=",k,"from previous site j=",j," at distance=",a*np.abs(k-j),"with G_eff:","%6.5f" % G_eff(R,a,j,k))

        #L.append(upD(j,N,G)*downD(k,N,G))

        scatterindex.append(j)

        #print("L",len(L)-1,": Transfer to scattering site j=",j,"from previous site k=",k,"at distance=",a*np.abs(k-j),"with G_eff:","%6.5f" % G_eff(R,a,j,k))
       
      if k!=j:

        g=np.sqrt(-1j*g_eff(R,a,j,k))

        #L.append(upD(k,N,g)*downD(k,N,g))

        scatterindex.append(j)

        #print("L",len(L)-1,": Scattering at j=",j,"due to impurity site k=",k,"at distance=",a*np.abs(k-j),"with g_eff:","%6.5f" % g_eff(R,a,j,k))

  print(L)
  return L



N=2
a=1
timesteps=100
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

for r in [4.0]:
  for t in np.ones(disavgs):
    print(i,"of",disavgs)
    i=i+1
    #print("Interaction Coefficient over one-half EIT bandwidth: ","%6.3f" % (r**(1/3)))
    #print("Interparticle spacing in units of [Critical Radius R]: ", "%6.3f" % (a/r))
    times = np.linspace(0.0, t*12.6, timesteps)
    results=[]
    asklist=[]

    for j in range(0,N):
      asklist.append(askatom(j,N))

    #result = mcsolve(H_eff(r,a,N) , productstate(0,N) , times, L_eff(r,a,N) , asklist, options=opts)

    result1 = mesolve(H(r,N,1) , productstate(0,N) , times, [] , [askatom(0,N)+askatom(1,N)], options=opts)

    result2 = mesolve(H_eff(r,a,N) , result1.states[timesteps-1] , np.linspace(0.0, t*10, timesteps), L_eff(r,a,N) , [askatom(0,N)+askatom(1,N)], options=opts)

    result3 = mesolve(H(r,N,1) , result2.states[timesteps-1] , times, [] , [askatom(0,N)+askatom(1,N)], options=opts)

    traceduu=[]
    traceddd=[]
    tracedbb=[]
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

      traceduu.append(np.abs(result2.states[t].ptrace(0)[0][0][0]))
      traceddd.append(np.abs(result2.states[t].ptrace(0)[1][0][1]))
      tracedbb.append(np.abs(result2.states[t].ptrace(0)[2][0][2]))

      #print(result1.states[t])

      #conc.append(concurrence(result.states[t]))
      VN.append(entropy_vn(result2.states[t]))

      #rhoA=result.states[t].ptrace(0)
      #rhoAsquare=rhoA*rhoA
      #purityA.append(rhoAsquare[0][0][0]+rhoAsquare[1][0][1])

      #rhoB=result.states[t].ptrace(1)
      #rhoBsquare=rhoB*rhoB
      #purityB.append(rhoBsquare[0][0][0]+rhoBsquare[1][0][1])

      #rhoC=result.states[t].ptrace(1)
      #rhoCsquare=rhoC*rhoC
      #purityC.append(rhoCsquare[0][0][0]+rhoCsquare[1][0][1])

    #purityAav=np.add(purityAav,purityA)
    #purityBav=np.add(purityBav,purityB)
    #purityCav=np.add(purityCav,purityC)
    #concav=np.add(concav,conc)
    VNav=np.add(VNav,VN)

    #print(bellstate(0,1,N))

    #print(H(r,N).eigenstates())

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
    #ax.plot(times, result.expect[0], label="n=%d"%(0,))

    #leg = plt.legend(loc='best', ncol=3, shadow=True, fancybox=True)
    #leg.get_frame().set_alpha(0.5)

    #ax.set_xlabel('Time');
    #ax.set_ylabel('Expectation value sigma(z)');
    #plt.show()

    #plt.savefig("evolution at t="+str(t*10)+" R_crit"+str(r)[0]+"."+str(r)[2]+".pdf", dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None, metadata=None)
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
ax.plot(times, np.abs(traceduu)/disavgs, label="rho_uu")
ax.plot(times, np.abs(traceddd)/disavgs, label="rho_dd")
ax.plot(times, np.abs(tracedbb)/disavgs, label="rho_bb")
#ax.plot(times, np.abs(purityAav)/disavgs, label="Purity_site_0")
#ax.plot(times, np.abs(purityBav)/disavgs, label="PurityB")
#ax.plot(times, np.abs(purityCav)/disavgs, label="PurityC")
#ax.plot(times, concav/disavgs, label="Concurrence")
ax.plot(times, VNav/disavgs, label="Von-Neumann Entropy")
leg = plt.legend(loc='upper right', ncol=1, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()
