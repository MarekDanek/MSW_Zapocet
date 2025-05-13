import numpy as np                      # Pro práci s poli a číselnými výpočty
from scipy.integrate import odeint     # Pro numerické řešení soustav diferenciálních rovnic
import matplotlib.pyplot as plt        # Pro vykreslování grafů

#1. SIR

# Celková velikost populace
N = 1000

# Počáteční stavy:
I0 = 1         # Počet nakažených na začátku
R0 = 0         # Počet uzdravených na začátku
S0 = N - I0 - R0  # Počet náchylných jedinců na začátku

# Časová osa – 300 dní s krokem 1 den
t = np.linspace(0, 300, 300)

# Mé vybrané nemoci
nemoci = [
    {"nazev": "Spalničky", "R0": 17, "doba_nemoci": 10},
    {"nazev": "Zarděnky", "R0": 6.5, "doba_nemoci": 10},
    {"nazev": "Chřipka", "R0": 3.5, "doba_nemoci": 10},
    {"nazev": "Tuberkulóza", "R0": 2.6, "doba_nemoci": 180},
    {"nazev": "Malárie", "R0": 100, "doba_nemoci": 10},
]

# Definice SIR modelu
def sir_model(y, t, beta, gamma):
    S, I, R = y
    # Míra změny počtu náchylných
    dSdt = -beta * S * I / N
    # Míra změny počtu nakažených
    dIdt = beta * S * I / N - gamma * I
    # Míra změny počtu uzdravených
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Vytvoření mřížky 3x2 pro více grafů v jednom okně
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs = axs.flatten()  # Převedení 2D pole os na 1D

# Cyklus přes všechny nemoci
for idx, nemoc in enumerate(nemoci):
    # Získání parametrů
    R0_val = nemoc["R0"]
    doba_nemoci = nemoc["doba_nemoci"]

    # Parametry modelu:
    gamma = 1 / doba_nemoci        # Rychlost uzdravení
    beta = R0_val * gamma          # Rychlost šíření infekce

    # Počáteční podmínky
    y0 = S0, I0, R0

    # Numerické řešení soustavy rovnic
    reseni = odeint(sir_model, y0, t, args=(beta, gamma)) # Vypocet dif. rovnic
    S, I, R = reseni.T  # Transpozice výsledku pro snadný přístup ke každé proměnné

    # Zjištění vrcholu epidemie – kdy je nejvíce nakažených
    vrchol_index = np.argmax(I)
    vrchol_cas = t[vrchol_index]

    # Délka epidemie – kdy počet nakažených klesne pod 1
    delka_epidemie = t[np.where(I < 1)[0][-1]] if np.any(I < 1) else t[-1]

    # Počet nakažených celkem = ti, co se nakonec uzdravili
    nakazeni_celkem = R[-1]
    zdravi_celkem = S[-1]

    # Výpis do konzole – pro pozdější popis
    print(f"\nNemoc: {nemoc['nazev']}")
    print(f" - Vrchol epidemie: den {vrchol_cas:.1f}")
    print(f" - Konec epidemie: den {delka_epidemie:.1f}")
    print(f" - Celkem nakazenych: {nakazeni_celkem:.0f}")
    print(f" - Zdravych zustane: {zdravi_celkem:.0f}")

    # Vykreslení do podgrafu
    ax = axs[idx]
    ax.plot(t, S, 'b', label='Náchylní (S)')
    ax.plot(t, I, 'r', label='Nakažení (I)')
    ax.plot(t, R, 'g', label='Uzdravení (R)')
    ax.axvline(x=vrchol_cas, color='r', linestyle='--', linewidth=1, label='Vrchol')
    ax.set_title(f"{nemoc['nazev']} (R₀ = {R0_val})")
    ax.set_xlabel("Čas (dny)")
    ax.set_ylabel("Počet lidí")
    ax.grid(True)
    ax.legend()

# Pokud je více polí než nemocí, zbylí se smažou
if len(nemoci) < len(axs):
    for i in range(len(nemoci), len(axs)):
        fig.delaxes(axs[i])

# # Zarovnání grafů a zobrazení
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # hspace = výška, wspace = šířka
plt.show()
plt.show()



#########################################################################################################

#2. model Lotka-Volterra


#základní verze se 2. druhy

# Parametry modelu
a = 1.1   # růst kořisti (x)
b = 0.4   # predace kořisti predátorem (x -> y)
c = 0.4   # úmrtnost predátora bez potravy (y)
d = 0.1  # přeměna kořisti na predátory (růst y)

# Počáteční stavy: kořist, predátor
x0 = 10  # kořist
y0 = 10   # predátor

# Časová osa
t = np.linspace(0, 50, 1000)

# Definice klasického Lotka-Volterra modelu pro 2 druhy
def lotka_volterra_2druhy(y, a, b, c, d, t):
    x, y_ = y
    x = max(x, 0)
    y_ = max(y_, 0)
    dxdt = a * x - b * x * y_          # růst kořisti - predace
    dydt = d * x * y_ - c * y_         # růst predátora - úmrtnost
    return dxdt, dydt

# Počáteční podmínky
y_init = [x0, y0]

# Řešení soustavy
reseni = odeint(lotka_volterra_2druhy, y_init, t, args=(a, b, c, d))
x, y_ = reseni.T

# Vykreslení výsledků
plt.figure(figsize=(12, 8))

plt.plot(t, x, 'g-', label="Kořist")
plt.plot(t, y_, 'r-', label="Predátor")

plt.title("Klasický Lotka-Volterra model (2 druhy)")
plt.xlabel("Čas")
plt.ylabel("Populace")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Výpis informací pro popis
print(f"- Počáteční stavy: x0={x0}, y0={y0}")
print("- Model ukazuje klasickou dynamiku predátor-kořist.")
print("- Dochází ke cyklickým výkyvům populací obou druhů v čase.")



# rozšířená verze s 3. druhem


# # Parametry modelu
# a = 0.6   # růst kořisti (x)
# b = 0.2    # predace kořisti predátorem (x -> y)
# c = 0.4    # úmrtnost predátora bez potravy (y)
# d = 0.05    # přeměna kořisti na predátory (růst y)
# e = 0.02   # predace predátora třetím druhem (y -> z)
# f = 0.02    # růst třetího druhu z potravy (y)

# # Počáteční stavy: kořist, predátor, 3. druh
# x0 = 10  # kořist
# y0 = 5  # predátor
# z0 = 3  #3. druh

# # # Časová osa 
# t = np.linspace(0, 200, 1000)

# # Definice rozšířeného Lotka-Volterra modelu se 3 druhy
# def lotka_volterra_3druhy(y, a, b, c, d, e, f, t):
#     x, y_, z = y  # rozbalení stavu
#     x = max(x, 0) # omezení růstu a záporných hodnot
#     y_ = max(y_, 0)
#     z = max(z, 0)
#     dxdt = a * x - b * x * y_            # růst kořisti - predace
#     dydt = d * x * y_ - c * y_ - e * y_ * z  # růst predátora - úmrtnost
#     dzdt = f * y_ * z                    # růst třetího druhu 
#     return dxdt, dydt, dzdt

# # Počáteční podmínky
# y_init = [x0, y0, z0]

# # Řešení soustavy
# reseni = odeint(lotka_volterra_3druhy, y_init, t, args=(a, b, c, d, e, f))
# x, y_, z = reseni.T  # transpozice výsledků

# # Vykreslení výsledků
# plt.figure(figsize=(12, 8))

# plt.plot(t, x, 'g-', label="Kořist")
# plt.plot(t, y_, 'r-', label="Predátor")
# plt.plot(t, z, 'b-', label="3. druh")

# plt.title("Rozšířený Lotka-Volterra model se třemi druhy")
# plt.xlabel("Čas")
# plt.ylabel("Populace")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Výpis informací pro popis
# print(f"- Pocatecni stavy: x0={x0}, y0={y0}, z0={z0}")
# print("- Model ukazuje, jak zavedeni 3. druhu ovlivnuje dynamiku mezi koristi a predatorem.")
# print("- Muze dojit ke kolapsu predatora, stabilizaci, nebo chaosu – zalezi na parametrech.")


#####################################################################################################
#3. Zombie apokalypsa


# # Definice modelu SZR – Susceptible, Zombies, Removed
# def szr_model(y, t, beta, delta, alpha, zeta, pi):
#     S, Z, R = y
#     dSdt = pi - beta * S * Z - delta * S
#     dZdt = beta * S * Z + zeta * R - alpha * S * Z
#     dRdt = delta * S + alpha * S * Z - zeta * R
#     return dSdt, dZdt, dRdt

# # Počáteční stavy: zdraví lidé, zombie, mrtví
# S0 = 500
# Z0 = 1
# R0 = 0
# y0 = [S0, Z0, R0]

# Časová osa 
# t = np.linspace(0, 50, 1000)

# # Parametry modelu
# beta = 0.005   # míra nakažení
# delta = 0.0001 # přirozená úmrtnost
# alpha = 0.005  # mrtví zombie
# zeta = 0.0001  # oživení mrtvých
# pi = 0         # porodnost

# # Výpočet řešení
# reseni = odeint(szr_model, y0, t, args=(beta, delta, alpha, zeta, pi))
# S, Z, R = reseni.T

# # Vizualizace výsledků
# plt.figure(figsize=(12, 6))
# plt.plot(t, S, label='Zdraví (S)', color='blue')
# plt.plot(t, Z, label='Zombie (Z)', color='red')
# plt.plot(t, R, label='Mrtví (R)', color='gray')
# plt.xlabel("Čas (dny)")
# plt.ylabel("Počet jedinců")
# plt.title("Zombie Apokalypsa – Model SZR")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



#rozšířená verze


# Rozšířený SZR model: infekce i z mrtvých těl a uzdravení zombie
def szr_healing_model(y, t, beta, delta, alpha, zeta, pi, gamma, eta):
    S, Z, R = y
    dSdt = pi - beta * S * Z - eta * S * R - delta * S + gamma * Z
    dZdt = beta * S * Z + eta * S * R + zeta * R - alpha * S * Z - gamma * Z
    dRdt = delta * S + alpha * S * Z - zeta * R
    return dSdt, dZdt, dRdt

# Časová osa
t = np.linspace(0, 100, 1000)

# Počáteční stavy: zdraví lidé, zombie, mrtví
S0 = 500
Z0 = 1
R0 = 0
y0 = [S0, Z0, R0]

# Parametry modelu
beta = 0.007     # míra nakažení od zombie
eta = 0.030      # míra nakažení od mrtvých
delta = 0.0001   # přirozená úmrtnost
alpha = 0.005    # mrtví po kontaktu se zombie
zeta = 0.0001    # oživení mrtvých
pi = 0           # porodnost (přírůstek lidí)
gamma = 0.4      # míra léčení zombie

# Řešení diferenciálních rovnic
reseni2 = odeint(szr_healing_model, y0, t, args=(beta, delta, alpha, zeta, pi, gamma, eta))
S2, Z2, R2 = reseni2.T

# Vizualizace výsledků
plt.figure(figsize=(12, 6))
plt.plot(t, S2, label='Zdraví (S)', color='blue')
plt.plot(t, Z2, label='Zombie (Z)', color='green')
plt.plot(t, R2, label='Mrtví (R)', color='red')
plt.xlabel("Čas (dny)")
plt.ylabel("Počet jedinců")
plt.title("SZR model s léčením a nakažením z mrtvých těl")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()