"main "

import numpy as np
from Simulate_Data import *
from risk_models import *
import matplotlib.pyplot as plt

# np.random.seed(23)

# da usare per quantile 1 - alpha 
alpha = 0.05 #0.10, 0.05, 0.01    
# Numero simulazioni
num_simulazioni = 20
#Numero scenari per simulazione
num_scenarios =5000 # 20, 50, 100, 200, 500, 1000, ...
# Scenari per numero studenti partecipanti (valori tra 10000 e 300000)
poss_num_students = np.arange(10000, 30001)
# Costo per ogni studente in euro
stud_price = 1000


# Parametri generali
# Tasso di interesse privo di rischio domestico 
rd = 0.03 
# Tasso di interesse privo di rischio estero
rf = 0.02
# Volatilità
volatility = 0.20 #0.10, 0.15, 0.20
# Maturità (1 anno)
T = 1  

# Prezzo spot attuale di cambio ($-> €)
S0 = 1.08
# Prezzo forward
F0 = S0 * np.exp(T*(rd - rf))
# Prezzi strike delle opzioni
K1 = 1.06
K2 = 1.08
K3 = 1.10
K = [K1, K2, K3]

# Calcolo dei prezzi delle opzioni call
Call_prices = []
for k in K:
    d1 = (np.log(S0 /k) + (rd - rf + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
    d2 = d1 - volatility * np.sqrt(T)
    C = S0 * np.exp(-rf * T) * norm.cdf(d1) - k * np.exp(-rd * T) * norm.cdf(d2)
    Call_prices.append(C)

[C1, C2, C3] = Call_prices

print(f"Prezzo della call con K1 = {K1}: {C1:.4f}")
print(f"Prezzo della call con K2 = {K2}: {C2:.4f}")
print(f"Prezzo della call con K3 = {K3}: {C3:.4f}")
print("#############################################################")
# Fattore di attualizzazione
att = np.exp(rd * T)

# Campioni del Cvar simulati
cvar_values = []
# Campioni del Var simulati
var_values = []
# Campioni della varianza simulati
variance_values = []
# Campioni dell' EVaR simulati
evar_values = []

# Lista di scenari identificati, dal numero di scenario
scenarios = [i for i in range(num_scenarios)]

#############################################################################################################

# Modelli
risk_models_optimizer = risk_models() 

# Misura di rischio scelta
risk_measure = "CVaR" #CVaR, VaR, varianza

# file dei log
log_file = f"simulation_log_{risk_measure}.txt"

# Scrittura nel file di log
with open(log_file, mode="w", newline="") as file:
    file.write(f"Simulazioni: Minimizzazione {risk_measure}\n")
    file.write("-------------------------------------------------------------\n")


   
# Ciclo su ogni simulazione
for j in range(num_simulazioni):
    print(f"Simulazione numero: {j + 1}")
    
    # Scrittura nel file di log
    with open(log_file, mode="a", newline="") as file:
        file.write(f"Simulazione numero: {j + 1}\n")
        
    # Dati ancora da simulare
    data_model = Simulate_Data(array_students = poss_num_students, distr ='Beta', S0 = S0, rf= rf, rd= rd, volatility= volatility, T = T ,num_scenarios=num_scenarios)
    
    # Numero studenti simulati in ogni scenario
    dataS = data_model.simulate_students()
                              
    # Prezzi spot futuri calcolati in ogni scenarii
    dataFutSpot = np.round(data_model.simulate_fut_Spot(), 5)
    

    # Fabbisogno in euro per ogni scenario di studenti (Volume)
    V = dataS * stud_price
    
    # Valore di riferimento
    R = V * S0 

    # Selezione e ottimizzazione del modello con la misura di rischio scelta
    
    if risk_measure == "CVaR":
        cvar_sim = risk_models_optimizer.CVaR_minimize(alpha,F0, att, K, Call_prices, dataFutSpot, V, R, num_scenarios, scenarios, log_file)
        # Aggiunta alla lista
        cvar_values.append(cvar_sim)
        
    elif risk_measure == "VaR":
        var_sim = risk_models_optimizer.VaR_minimize(alpha,F0, att, K, Call_prices, dataFutSpot, V, R, num_scenarios, scenarios, log_file)
        # Aggiunta alla lista
        var_values.append(var_sim)
        
    elif risk_measure == "EVaR":
        evar_sim = risk_models_optimizer.EVaR_minimize(alpha,F0, att, K, Call_prices, dataFutSpot, V, R, num_scenarios, scenarios, log_file)
        # Aggiunta alla lista
        evar_values.append(evar_sim)

    elif risk_measure == "varianza":        
        variance_sim = risk_models_optimizer.variance_minimize(
    alpha, F0, att, K, Call_prices, dataFutSpot, V, R, num_scenarios, scenarios, log_file)
        # Aggiunta alla lista
        variance_values.append(variance_sim)


print('Risultati con ',num_scenarios, 'scenari, ', 'alpha: ', alpha, 'volatility; ', volatility)

# Calcolo media, deviazione standard e coefficiente di variazione percentuale dei risultati

if risk_measure == "CVaR":
    mean_cvar = np.mean(cvar_values)
    std_cvar = np.std(cvar_values)
    cv_cvar = std_cvar / mean_cvar * 100
    print(f"La media del CVaR è: {mean_cvar:.2f}.\nIl coefficiente di variazione percentuale è: {cv_cvar:.3f}%.")
    
elif risk_measure == "VaR":
    mean_var = np.mean(var_values)
    std_var = np.std(var_values)
    cv_var = std_var / mean_var * 100
    print(f"La media del VaR è : {mean_var:.2f}.\nIl coefficiente di variazione percentuale è {cv_var:.3f}%.")
    
elif risk_measure == "EVaR":
    mean_evar = np.mean(evar_values)
    std_evar = np.std(evar_values)
    cv_evar = std_evar / mean_evar * 100
    print(f"La media del EVaR è : {mean_evar:.2f}.\nIl coefficiente di variazione percentuale è {cv_evar:.3f}%.")
    
elif risk_measure == "varianza":
    std_values = np.sqrt(variance_values)
    mean_std = np.mean(std_values)
    std_std = np.std(std_values)
    cv_std = std_std / mean_std * 100
    print(f"La media della deviazione standard delle loss è : {mean_std:.2f}.\nIl coefficiente di variazione percentuale è {cv_std:.3f}%.")   
   # mean_varianza = np.mean(variance_values)
   # std_varianza = np.std(variance_values)
   # cv_varianza = std_varianza / mean_varianza * 100
   # print(f"La media della varianza è : {mean_varianza:.2f}.\nIl coefficiente di variazione percentuale è {cv_varianza:.3f}%.")


    


"""
# Barplot simulazioni Cvar
label = "Cvar"
data = cvar_values
x_vals = np.linspace(min(data) - 10000 * abs(min(data)), 
                     max(data) + 10000 * abs(max(data)), 1000)
numero_bins = 100
plt.figure(figsize=(10, 6))
plt.hist(data, density=True, alpha=0.5,bins= numero_bins, label=f'Istogramma dei campioni ({label})')
plt.xlabel('Valori')

plt.axvline(np.mean(data), color='red', linestyle='--', label=f'Media del {label}')
plt.title(f'Risultati di {num_simulazioni} simulazioni con {num_scenarios} scenari del {label}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.show()

mean = np.mean(data)
print(f"La media del {label} è {mean:.4f}")"""
