import numpy as np
import matplotlib.pyplot as plt

# Mixing angles in radians
t12 = np.deg2rad(33.41)
t13 = np.deg2rad(8.54)
t23 = np.deg2rad(49.1)

# Mass-squared differences (eV^2)
delta12 = 7.37e-5
delta13 = 2.56e-3
delta23 = delta13 - delta12

# Baseline and L/E
L = 1300  # km
E = np.logspace(-2,np.log10(30), 1001)
LbyE = L/E

# CP phases (0 to 200 step 40)
CP_degrees = [0,45,90,180,217]
CP_phases = np.deg2rad(CP_degrees)


for CP in CP_phases:  # start index at 1
    def U_matrix(CP,t12,t13,t23):
        s12, s23, s13, sCP = np.sin(t12), np.sin(t23), np.sin(t13), np.sin(CP)
        c12, c23, c13, cCP = np.cos(t12), np.cos(t23), np.cos(t13), np.cos(CP)
        
        U=np.array([
            [c12*c13,                  s12*c13,               s13*np.exp(-1j*CP)],
            [-s12*c23-c12*s23*s13*np.exp(1j*CP),               c12*c23-s12*s23*s13*np.exp(1j*CP),               s23*c13],
            [s12*s23-c12*c23*s13*np.exp(1j*CP),               -c12*s23-s12*c23*s13*np.exp(1j*CP),              c23*c13]
            ],dtype=complex)
        
        return U
    U = U_matrix(CP, t12, t13, t23)
    
    U1_real = (U[0][1].conj()*U[1][1]*U[0][0]*U[1][0].conj()).real 
    U2_real = (U[0][2].conj()*U[1][2]*U[0][1]*U[1][1].conj()).real
    U3_real = (U[0][2].conj()*U[1][2]*U[0][0]*U[1][0].conj()).real

    # Imaginary parts
    U1_img = (U[0][1].conj()*U[1][1]*U[0][0]*U[1][0].conj()).imag
    U2_img = (U[0][2].conj()*U[1][2]*U[0][1]*U[1][1].conj()).imag
    U3_img = (U[0][2].conj()*U[1][2]*U[0][0]*U[1][0].conj()).imag

#FOR THE OSCILLATION PROBABILITY OF ELECTRON TO MUON NEUTRINOS:
    Osc_Probability_mu = (
        -4 * (U1_real * (np.sin(1.27 * delta12 * LbyE))**2 +
              U2_real * (np.sin(1.27 * delta23 * LbyE))**2 +
              U3_real * (np.sin(1.27 * delta13 * LbyE))**2)
        + 2 * (U1_img * np.sin(2.54 * delta12 * LbyE) +
               U2_img * np.sin(2.54 * delta23 * LbyE) +
               U3_img * np.sin(2.54 * delta13 * LbyE))
    )

#FOR THE SURVIVAL PROBABILITY OF ELCTRON NEUTRINOS:
    U1_sur = (U[0][0]*U[0][1].conj()*U[0][0].conj()*U[0][1]).real
    U2_sur = (U[0][0]*U[0][2].conj()*U[0][0].conj()*U[0][2]).real
    U3_sur = (U[0][1]*U[0][2].conj()*U[0][1].conj()*U[0][2]).real

    Sur_Probability = 1 - 4 * (
    U1_sur * np.sin(1.27 * delta12 * LbyE)**2 +
    U2_sur * np.sin(1.27 * delta13 * LbyE)**2 +
    U3_sur * np.sin(1.27 * delta23 * LbyE)**2)

#FOR THE OSCILLATION PROBABILITY OF ELECTRON TO TAU NEUTRINOS:
    
    U1_real_tau=(U[0][1].conj()*U[2][1]*U[0][0]*U[2][0].conj()).real

    U2_real_tau=(U[0][2].conj()*U[2][2]*U[0][1]*U[2][1].conj()).real
    U3_real_tau=(U[0][2].conj()*U[2][2]*U[0][0]*U[2][0].conj()).real

    U1_img_tau=(U[0][1].conj()*U[2][1]*U[0][0]*U[2][0].conj()).imag
    U2_img_tau=(U[0][2].conj()*U[2][2]*U[0][1]*U[2][1].conj()).imag
    U3_img_tau=(U[0][2].conj()*U[2][2]*U[0][0]*U[2][0].conj()).imag
    # Oscillation probability
    Osc_Probability_tau= (
        -4 * (U1_real_tau * (np.sin(1.27 * delta12 * LbyE))**2 +
              U2_real_tau * (np.sin(1.27 * delta23 * LbyE))**2 +
              U3_real_tau * (np.sin(1.27 * delta13 * LbyE))**2)
        + 2 * (U1_img_tau * np.sin(2.54 * delta12 * LbyE) +
               U2_img_tau * np.sin(2.54 * delta23 * LbyE) +
               U3_img_tau * np.sin(2.54 * delta13 * LbyE))
    )
    
    
    plt.ylim(0,1)
    plt.xscale('log')
    plt.xlabel("E (MeV)")
    plt.plot(E, Osc_Probability_mu, label="$\\nu_{e} \\rightarrow \\nu_{\\mu}$")
    plt.plot(E, Sur_Probability,label="$\\nu_{e} \\rightarrow \\nu_{e}$")
    plt.plot(E, Osc_Probability_tau,label="$\\nu_{e} \\rightarrow \\nu_{\\tau}$")
    plt.title(f'CP Phase = {np.rad2deg(CP)}°')
    plt.legend()
    plt.grid(True)
    plt.show()

            

