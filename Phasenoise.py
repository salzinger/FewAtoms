import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check

opts = Options(store_states=True, store_final_state=True, ntraj=200)


def twobasis():
    return np.array([basis(2, 0), basis(2, 1)], dtype=object)


def productstateZ(up_atom, down_atom, N):
    up, down = twobasis()
    #pbasis = np.full(N, down)
    pbasis = up
    #pbasis[down_atom] = down
    return tensor(pbasis)

def productstateX(m, j, N):
    up, down = twobasis()
    #pbasis = np.full(N, down)
    pbasis = (up + down).unit()
    #pbasis[j] = (up + down).unit()
    return tensor(pbasis)


def sigmap(m, N):
    up, down = twobasis()
    #oplist = np.full(N, identity(2))
    oplist = up * down.dag()
    return tensor(oplist)


def sigmam(m, N):
    up, down = twobasis()
    #oplist = np.full(N, identity(2))
    oplist = down * up.dag()
    return tensor(oplist)


def sigmaz(j, N):
    #oplist = np.full(N, identity(2))
    oplist = Qobj([[1, 0], [0, -1]])
    return tensor(oplist)


def sigmax(j, N):
    #oplist = np.full(N, identity(2))
    oplist = Qobj([[0, 1], [1, 0]])
    return tensor(oplist)


def sigmay(j, N):
    #oplist = np.full(N, identity(2))
    oplist = Qobj([[0, -1j], [1j, 0]])
    return tensor(oplist)


def MagnetizationZ(N):
    sum = 0
    for j in range(0, N):
        sum += sigmaz(j, N)
    return sum / N


def MagnetizationX(N):
    sum = 0
    for j in range(0, N):
        sum += sigmax(j, N)
    return sum / N


def MagnetizationY(N):
    sum = 0
    for j in range(0, N):
        sum += sigmay(j, N)
    return sum / N


def H0(omega, J, N):
    H = 0
    for j in range(0, N):
        H += 1 * omega / 2 * sigmaz(j, N)
        for i in range(0, N):
            if i != j:
                H += J * (sigmax(i, N) * sigmax(j, N) + sigmay(i, N) * sigmay(j, N))
    return H


def H1(Omega_R, N):
    H = 0
    for j in range(0, N):
        H -= Omega_R * (sigmap(j, N))
    return H


def H2(Omega_R, N):
    H = 0
    for j in range(0, N):
        H -= Omega_R * (sigmam(j, N))
    return H


N = 1

omega = 2. * np.pi * 20

Omega_R = 2. * np.pi * 4

J = 1

timesteps = 400

endtime = 2
pertubation_length = endtime/1

t1 = np.linspace(0, endtime, timesteps)
t2 = np.linspace(0, endtime, timesteps)

perturb_times = np.linspace(0, pertubation_length, timesteps)
random_phase = 0.00 * np.random.randn(perturb_times.shape[0])

func1 = lambda t: 0.5j*np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
noisy_func1 = lambda t: func1(t + random_phase)
noisy_data1 = noisy_func1(perturb_times)
S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data1)

func2 = lambda t: 0.5j*np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
noisy_func2 = lambda t: func2(t + random_phase)
noisy_data2 = noisy_func2(perturb_times)
S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(0, N)]

Perturb = sigmax(0, N)
Measure = sigmay(0, N)

opts = Options(store_states=True, store_final_state=True, ntraj=50)

result_t1 = mcsolve(H0(omega, J, N), productstateZ(0, 0, N), t1, [Qobj([[np.exp(1j), 0], [0, np.exp(1j)]])], Exps, options=opts,
                    progress_bar=True)

result_t1t2 = mcsolve(H0(omega, J, N), result_t1.states[:, timesteps-1].mean().unit(), t2, [],
                      Exps, options=opts,
                      progress_bar=True)

result_AB = mcsolve(H0(omega, J, N), Perturb * result_t1.states[:, timesteps-1].mean().unit(), t2, [],
                    Exps, options=opts,
                    progress_bar=True)

prod_AB = result_t1t2.states[timesteps - 1].dag() * Measure * result_AB.states[timesteps - 1]

prod_BA = result_AB.states[timesteps - 1].dag() * Measure * result_t1t2.states[timesteps - 1]

Commutator = prod_AB - prod_BA

AntiCommutator = prod_AB + prod_BA

result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], result_t1.states[:, timesteps-1].mean().unit(),
                  perturb_times, [], Exps, options=opts, progress_bar=True)

result3 = mesolve(H0(omega, J, N), result2.states[timesteps - 1], t2, [],
                  Exps, options=opts,
                  progress_bar=True)

print('Initial state ....')
print(productstateZ(0, 0, N))
print(productstateZ(0, 0, N).dag()*sigmaz(1, N)*productstateZ(0, 0, N))


print('H0...')
print(H0(omega, J, N))
print('H1...')
print(H1(Omega_R, N))
print('H2...')
print(H2(Omega_R, N))

print('Commutator:', 1j * Commutator[0][0])
print('AntiCommutator: ', AntiCommutator[0][0])

print(result_t1.states[:, timesteps-1].mean().unit())

fig, ax = plt.subplots(3, 3)
ax[0, 0].plot(perturb_times, func1(perturb_times))
ax[0, 0].plot(perturb_times, noisy_data1, 'o')
ax[0, 0].plot(perturb_times, S1(perturb_times), lw=2)
ax[0, 0].set_xlabel('Time [1/J]')
ax[0, 0].set_ylabel('Coupling Amplitude')
ax[0, 0].set_xlim([0, 0.1])

ax[0, 1].plot(perturb_times, S1(perturb_times), lw=2)
ax[0, 1].set_xlabel('Time [1/J]')

ax[0, 2].plot(perturb_times, func2(perturb_times))
ax[0, 2].plot(perturb_times, noisy_data2, 'o')
ax[0, 2].plot(perturb_times, S2(perturb_times), lw=2)
ax[0, 2].set_xlabel('Time [1/J]')
ax[0, 2].set_xlim([0, 0.1])


ax[1, 0].plot(t1, result_t1.expect[0], label="MagnetizationX")
ax[1, 0].plot(t1, result_t1.expect[1], label="MagnetizationZ")
ax[1, 0].plot(t1, result_t1.expect[2], label="MagnetizationY")
#ax[1, 0].plot(t1, result_t1.expect[3], label="tensor(SigmaZ,Id) ")
#ax[1, 0].plot(t1, result_t1.expect[4], label="tensor(Id,SigmaZ) ")
ax[1, 0].set_xlabel('Free Evolution Time [1/J]')
ax[1, 0].set_ylabel('Magnetization')
ax[1, 0].legend(loc="upper right")
ax[1, 0].set_ylim([-1.1, 1.1])

ax[1, 1].plot(perturb_times, result2.expect[0], label="MagnetizationX")
ax[1, 1].plot(perturb_times, result2.expect[1], label="MagnetizationZ")
ax[1, 1].plot(perturb_times, result2.expect[2], label="MagnetizationY")
#ax[1, 1].plot(perturb_times, result2.expect[3], label="tensor(SigmaZ,Id) ")
#ax[1, 1].plot(perturb_times, result2.expect[4], label="tensor(Id,SigmaZ) ")
ax[1, 1].set_xlabel('Perturbation Time [1/J]')
ax[1, 1].legend(loc="right")
ax[1, 1].set_ylim([-1.1, 1.1])

ax[1, 2].plot(t2, result3.expect[0], label="MagnetizationX")
ax[1, 2].plot(t2, result3.expect[1], label="MagnetizationZ")
ax[1, 2].plot(t2, result3.expect[2], label="MagnetizationY")
#ax[1, 2].plot(t2, result3.expect[3], label="tensor(SigmaZ,Id) ")
#ax[1, 2].plot(t2, result3.expect[4], label="tensor(Id,SigmaZ) ")
ax[1, 2].set_xlabel('Free Evolution time [1/J]')
ax[1, 2].legend(loc="right")
ax[1, 2].set_ylim([-1.1, 1.1])


ax[2, 0].plot(t1, result_t1.expect[0], label="MagnetizationX")
ax[2, 0].plot(t1, result_t1.expect[1], label="MagnetizationZ")
ax[2, 0].plot(t1, result_t1.expect[2], label="MagnetizationY")
#ax[2, 0].plot(t1, result_t1.expect[3], label="tensor(SigmaZ,Id) ")
#ax[2, 0].plot(t1, result_t1.expect[4], label="tensor(Id,SigmaZ) ")
ax[2, 0].set_xlabel('Free Evolution time [1/J]')
ax[2, 0].legend(loc="right")
ax[2, 0].set_ylim([-1.1, 1.1])

ax[2, 1].plot(t2, result_AB.expect[0], label="MagnetizationX")
ax[2, 1].plot(t2, result_AB.expect[1], label="MagnetizationZ")
ax[2, 1].plot(t2, result_AB.expect[2], label="MagnetizationY")
#ax[2, 1].plot(t2, result_AB.expect[3], label="tensor(SigmaZ,Id)")
#ax[2, 1].plot(t2, result_AB.expect[4], label="tensor(Id,SigmaZ)")
ax[2, 1].set_xlabel('After Perturbation [1/J]')
ax[2, 1].legend(loc="right")
ax[2, 1].set_ylim([-1.1, 1.1])

ax[2, 2].plot(t2, result_t1t2.expect[0], label="MagnetizationX")
ax[2, 2].plot(t2, result_t1t2.expect[1], label="MagnetizationZ")
ax[2, 2].plot(t2, result_t1t2.expect[2], label="MagnetizationY")
#ax[2, 2].plot(t2, result_t1t2.expect[3], label="tensor(SigmaZ,Id)")
#ax[2, 2].plot(t2, result_t1t2.expect[4], label="tensor(Id,SigmaZ)")
ax[2, 2].set_xlabel('No Perturbation [1/J]')
ax[2, 2].legend(loc="right")
ax[2, 2].set_ylim([-1.1, 1.1])

plt.show()
