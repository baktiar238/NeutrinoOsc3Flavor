import numpy as np
import matplotlib.pyplot as plt

#Neutrino Oscillation parameters
dm21 = 7.5e-5       # eV^2
dm31 = 2.5e-3       # eV^2
t12 = np.deg2rad(33.2)
t13 = np.deg2rad(8.6)
t23 = np.deg2rad(46.1)
deltaCP_list = np.deg2rad([0,45,90,135,180,217])

#Constants for unit conversions
E_GeV_2_eV = 1e9
km_2_eV = 5.06773e9  # 1 km = 5.06773×10^9 eV^-1

#L and L/E setup
L = 1300  # km
E = np.logspace(-2,np.log10(30), 1001)
L_evinv = L * km_2_eV

#PMNS matrix
def U_pmns(t12, t13, t23, CP):
    c12, s12 = np.cos(t12), np.sin(t12)
    c13, s13 = np.cos(t13), np.sin(t13)
    c23, s23 = np.cos(t23), np.sin(t23)
    e_CP = np.exp(-1j * CP)

    return np.array([
        [c12*c13,                     s12*c13,                     s13*e_CP],
        [-s12*c23 - c12*s23*s13*e_CP,  c12*c23 - s12*s23*s13*e_CP,  s23*c13],
        [ s12*s23 - c12*c23*s13*e_CP, -c12*s23 - s12*c23*s13*e_CP,  c23*c13]
    ], dtype=complex)

for l in range (0,L):
    def fn_rho(l):
        if l < 200:
            return 2.7
        else:
            return 2.9

rho = fn_rho(L)
Ye = 0.5
V = 7.56e-14 * rho * Ye

#Main loop over CP phases
for CP in deltaCP_list:
    U = U_pmns(t12, t13, t23, CP)
    P_ue_vals, P_uu_vals, P_ut_vals = [], [], []

    for Ei in E:
        E_eV = Ei * E_GeV_2_eV
        H_f_vac = U @ np.diag([0, dm21/(2*E_eV), dm31/(2*E_eV)]) @ U.conjugate().T
        H_f_matter = H_f_vac + np.diag([V, 0, 0])
        
#Computational results:
        evals, evecs = np.linalg.eigh(H_f_matter)
        W, lambdas = evecs, evals

        phase = np.exp(-1j * lambdas * L_evinv)
        S = W @ np.diag(phase) @ W.conjugate().T
        
#My analytical way:
        p = -np.trace(H_f_matter)
        q = 0.5 * (np.trace(H_f_matter)**2 - np.trace(H_f_matter@H_f_matter))
        r = -np.linalg.det(H_f_matter)

# Depressed cubic: x = λ + p/3  →  x^3 + a x + b = 0
        a = q - (p**2)/3
        b = (2*p**3)/27 - (p*q)/3 + r

# Discriminant terms
        delta = (b/2)**2 + (a/3)**3

# Cube roots (principal complex cube roots)
        C_plus  = (-b/2 + np.sqrt(delta))**(1/3)
        C_minus = (-b/2 - np.sqrt(delta))**(1/3)

# Cube roots of unity
        omega = np.exp(1j * 2*np.pi/3)

# Three roots (lamda_j)
        lamda = []
        for k in range(3):
            xj = C_plus * omega**(k) + C_minus * omega**(-k)
            lamdaj = xj - p/3  # undo shift (since p = -Tr(H))
            lamda.append(lamdaj.real)

# probabilities (note: indices are S[beta, alpha])
        A_ue = S[0, 1]
        A_uu = S[1, 1]
        A_ut = S[2, 1]

        P_ue_vals.append(np.abs(A_ue)**2)
        P_uu_vals.append(np.abs(A_uu)**2)
        P_ut_vals.append(np.abs(A_ut)**2)
        
# Probability expression:
    P_ue_vals = np.array(P_ue_vals)
    P_uu_vals = np.array(P_uu_vals)
    P_ut_vals = np.array(P_ut_vals)

# Plotting:

    print("lamdas (analytical)=",sorted(lamda))
    print("lamdas (computational)=",lambdas)
    plt.ylim(0,1)
    plt.xlim(0.5,30)
    plt.xscale('log')
    plt.plot(E, P_ue_vals, label="$\\nu_{\\mu} \\rightarrow \\ \\nu_{e}$")
    plt.plot(E, P_uu_vals, label="$\\nu_{\\mu} \\rightarrow \\ \\nu_{\\mu}$")
    plt.plot(E, P_ut_vals, label="$\\nu_{\\mu} \\rightarrow \\ \\nu_{\\tau}$")
    plt.xlabel(r"$E$ (GeV)")
    plt.ylabel("Probability")
    plt.title(f"Probability in Matter ($\\delta$ = {np.rad2deg(CP)}°)")
    plt.legend()
    plt.grid(True)
    plt.show()

