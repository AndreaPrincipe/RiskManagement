import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
class risk_models:

    # Modello per la mimizzazione del CVaR
    def CVaR_minimize(self, alpha, F0, att, K, Call_prices, dataFutSpot, V, R, num_scenarios, scenarios, log_file):
        model = gp.Model("Minimizzazione_CVaR")

        # Variabili
        K1, K2, K3 = K
        C1, C2, C3 = Call_prices
        
        # Variabili del modello
        VaR = model.addVar(name="VaR", lb=0)
        CVaR = model.addVar(name="CVaR", lb=0)
        loss = model.addVars(scenarios, name="loss", lb=0) 
        
        # Variabili ausiliarie per max(0, loss_s - VaR)
        gamma = model.addVars(scenarios, name="gamma", lb=0) 
        # Variabili delle call comprate
        y = model.addVars(3, name="y", lb=0)  
        # Quanto contratti forward comprati
        x = model.addVar(name="x", lb=0)  
        # Flusso di cassa in uscita
        O = model.addVars(scenarios, name="O", lb=0)  
        # z_s - w_s: quanto comprare/vendere spot
        zw = model.addVars(scenarios, name="zw", lb=-GRB.INFINITY)  
        # Opzioni call esercitate
        h = model.addVars(scenarios, 3, name="h", lb=0)  
        # Variabile binaria : 1 se loss_s <= Var, 0 altrimenti
        loss_lower_VaR = model.addVars(scenarios, vtype=GRB.BINARY, name="loss_lower_VaR")
        # Grande M
        M = 1e8  
        
        # Vincoli
        for i in scenarios:
            # Vincolo bilancio flusso in uscita
            model.addConstr(O[i] == x * F0 + zw[i] * dataFutSpot[i] + att * (C1 * y[0] + C2 * y[1] + C3 * y[2]) + K1 * h[i, 0] + K2 * h[i, 1] + K3 * h[i, 2], name=f"cash_flow_{i}")
            # Vincolo di fabbisogno
            model.addConstr(x + h[i, 0] + h[i, 1] + h[i, 2] + zw[i] == V[i], name=f"balance_{i}")
            # Vincoli su call esercitate
            model.addConstr(h[i, 0] <= y[0], name=f"h1_constraint_{i}")
            model.addConstr(h[i, 1] <= y[1], name=f"h2_constraint_{i}")
            model.addConstr(h[i, 2] <= y[2], name=f"h3_constraint_{i}")
            # gamma[i] = max(0, loss[i] - VaR)
            model.addConstr(gamma[i] >= loss[i] - VaR, name=f"gamma")
            # loss[i] = max(0, O[i] - R[i])
            model.addConstr(loss[i] >= O[i] - R[i], name=f"loss_lower_bound_{i}")  
            
            # Vincoli per il VaR
            model.addConstr(-M * (1 - loss_lower_VaR[i]) <= VaR - loss[i])
            model.addConstr(VaR - loss[i] <= M * (loss_lower_VaR[i]))
        
        # VaR definito come  1 - alpha quantile
        model.addConstr(gp.quicksum(loss_lower_VaR[i] / num_scenarios for i in scenarios) >= 1 - alpha)
        
        # Definizione del CVaR
        model.addConstr(CVaR == VaR + (1 / (num_scenarios * alpha)) * gp.quicksum(gamma[i] for i in scenarios), name="CVaR_definition")
        
        # Funzione obiettivo
        model.setObjective(CVaR, GRB.MINIMIZE)

        model.setParam('OutputFlag', 0)  
        
        # Ottimizzazione del modello
        start_time = time.time()  # Tempo iniziale
        model.optimize()  # Ottimizzazione del modello
        end_time = time.time()  # Tempo finale
        execution_time = end_time - start_time
        
        
        if model.status == GRB.OPTIMAL:
            cvar_value = CVaR.X
            x_value = x.X
            y_values = [y[i].X for i in range(3)]
            
            # Salvataggio dei dati nel file log
            with open(log_file, mode="a") as file:
                file.write(f"CVaR: {cvar_value:.4f}\n")
                file.write(f"x: {x_value:.4f}\n")
                file.write(f"y1: {y_values[0]:.4f}\n")
                file.write(f"y2: {y_values[1]:.4f}\n")
                file.write(f"y3: {y_values[2]:.4f}\n")
                file.write(f"Tempo di esecuzione: {execution_time:.4f} secondi\n")
                file.write("-------------------------------------------------------------\n")
            
            print(f"CVaR: {cvar_value:.4f}")
            print(f"x: {x_value:.4f}")
            print(f"y1: {y_values[0]:.4f}")
            print(f"y2: {y_values[1]:.4f}")
            print(f"y3: {y_values[2]:.4f}")
            print(f"Tempo di esecuzione: {execution_time:.4f} secondi")
            print("-------------------------------------------------------------")
            
            return cvar_value
        else:
            print("Ottimizzazione non riuscita.")
        
            return None


    # Modello per la mimizzazione del VaR
    def VaR_minimize(self, alpha, F0, att, K, Call_prices, dataFutSpot, V, R, num_scenarios, scenarios, log_file):
        model = gp.Model("Minimizzazione_VaR")

        # Variabili
        K1, K2, K3 = K
        C1, C2, C3 = Call_prices
        # Variabili del modello
        VaR = model.addVar(name="VaR", lb=0)
        loss = model.addVars(scenarios, name="loss", lb=0) 
        y = model.addVars(3, name="y", lb=0)  
        x = model.addVar(name="x", lb=0)  
        O = model.addVars(scenarios, name="O", lb=0)  
        zw = model.addVars(scenarios, name="zw", lb=-GRB.INFINITY)  
        h = model.addVars(scenarios, 3, name="h", lb=0)  
        loss_lower_VaR = model.addVars(scenarios, vtype=GRB.BINARY, name="loss_lower_VaR")
        # Variabili binarie per calcolo loss
        z = model.addVars(scenarios, vtype=GRB.BINARY, name="z")
  
        M = 1e8  
        
        # Vincoli
        for i in scenarios:
            model.addConstr(O[i] == x * F0 + zw[i] * dataFutSpot[i] + att * (C1 * y[0] + C2 * y[1] + C3 * y[2]) + K1 * h[i, 0] + K2 * h[i, 1] + K3 * h[i, 2], name=f"cash_flow_{i}")
            model.addConstr(x + h[i, 0] + h[i, 1] + h[i, 2] + zw[i] == V[i], name=f"balance_{i}")
            model.addConstr(h[i, 0] <= y[0], name=f"h1_constraint_{i}")
            model.addConstr(h[i, 1] <= y[1], name=f"h2_constraint_{i}")
            model.addConstr(h[i, 2] <= y[2], name=f"h3_constraint_{i}")
            # Vincoli per  loss[i] = max(0, O[i] - R[i]), serve grande M in questo caso 
            model.addConstr(loss[i] <= z[i] * M )  
            model.addConstr(loss[i] >= O[i] - R[i] )  
            model.addConstr(loss[i] <= O[i] - R[i] + (1 - z[i]) * M)
            #model.addConstr(loss[i] == O[i] ) 
            # Vincoli per il VaR
            model.addConstr(-M * (1 - loss_lower_VaR[i]) <= VaR - loss[i] )
            model.addConstr(VaR - loss[i] <= M * (loss_lower_VaR[i]) )

        # VaR definito come  1 - alpha quantile
        model.addConstr(gp.quicksum(loss_lower_VaR[i] / num_scenarios for i in scenarios) >= 1 - alpha)

        # Funzione obiettivo
        model.setObjective(VaR, GRB.MINIMIZE)
        
        model.setParam('OutputFlag', 0) 
        
        
        # Ottimizzazione del modello
        start_time = time.time()  
        model.optimize()  
        end_time = time.time()  
        execution_time = end_time - start_time
        
        if model.status == GRB.OPTIMAL:
            var_value = VaR.X
            x_value = x.X
            y_values = [y[i].X for i in range(3)]
            
            # Salvataggio dei dati nel file log
            with open(log_file, mode="a") as file:
                file.write(f"VaR: {var_value:.4f}\n")
                file.write(f"x: {x_value:.4f}\n")
                file.write(f"y1: {y_values[0]:.4f}\n")
                file.write(f"y2: {y_values[1]:.4f}\n")
                file.write(f"y3: {y_values[2]:.4f}\n")
                file.write(f"Tempo di esecuzione: {execution_time:.4f} secondi\n")
                file.write("-------------------------------------------------------------\n")

            print(f"VaR: {var_value:.4f}")
            print(f"x: {x_value:.4f}")
            print(f"y1: {y_values[0]:.4f}")
            print(f"y2: {y_values[1]:.4f}")
            print(f"y3: {y_values[2]:.4f}")
            print(f"Tempo di esecuzione: {execution_time:.4f} secondi")
            print("-------------------------------------------------------------")
            
            return var_value
        else:
            print("Ottimizzazione non riuscita.")
            return None
        
        return var_value
    
    
    # Modello per la mimizzazione del VaR con loss semplificata : loss[i] = O[i]
    def VaR_minimize_2(self, alpha, F0, att, K, Call_prices, dataFutSpot, V, R, num_scenarios, scenarios, log_file):
        model = gp.Model("Minimizzazione_VaR")

        # Variabili
        K1, K2, K3 = K
        C1, C2, C3 = Call_prices
        # Variabili del modello
        VaR = model.addVar(name="VaR", lb=0)
        y = model.addVars(3, name="y", lb=0)  
        x = model.addVar(name="x", lb=0)  
        O = model.addVars(scenarios, name="O", lb=0)  
        zw = model.addVars(scenarios, name="zw", lb=-GRB.INFINITY)  
        h = model.addVars(scenarios, 3, name="h", lb=0)  
        loss_lower_VaR = model.addVars(scenarios, vtype=GRB.BINARY, name="loss_lower_VaR")
    
        M = 1e8  
        
        # Vincoli
        for i in scenarios:

            model.addConstr(O[i] == x * F0 + zw[i] * dataFutSpot[i] + att * (C1 * y[0] + C2 * y[1] + C3 * y[2]) + K1 * h[i, 0] + K2 * h[i, 1] + K3 * h[i, 2], name=f"cash_flow_{i}")
            model.addConstr(x + h[i, 0] + h[i, 1] + h[i, 2] + zw[i] == V[i], name=f"balance_{i}")
            model.addConstr(h[i, 0] <= y[0], name=f"h1_constraint_{i}")
            model.addConstr(h[i, 1] <= y[1], name=f"h2_constraint_{i}")
            model.addConstr(h[i, 2] <= y[2], name=f"h3_constraint_{i}")

            # Vincoli per il VaR
            model.addConstr(-M * (1 - loss_lower_VaR[i]) <= VaR - O[i] )
            model.addConstr(VaR - O[i] <= M * (loss_lower_VaR[i]) )

        # VaR definito come  1 - alpha quantile
        model.addConstr(gp.quicksum(loss_lower_VaR[i] / num_scenarios for i in scenarios) >= 1 - alpha)

        # Funzione obiettivo
        model.setObjective(VaR, GRB.MINIMIZE)
        
        model.setParam('OutputFlag', 0) 
        
        
        # Ottimizzazione del modello
        start_time = time.time()  
        model.optimize()  
        end_time = time.time()  
        execution_time = end_time - start_time
        
        if model.status == GRB.OPTIMAL:
            var_value = VaR.X
            x_value = x.X
            y_values = [y[i].X for i in range(3)]
            
            # Salvataggio dei dati nel file log
            with open(log_file, mode="a") as file:
                file.write(f"VaR: {var_value:.4f}\n")
                file.write(f"x: {x_value:.4f}\n")
                file.write(f"y1: {y_values[0]:.4f}\n")
                file.write(f"y2: {y_values[1]:.4f}\n")
                file.write(f"y3: {y_values[2]:.4f}\n")
                file.write(f"Tempo di esecuzione: {execution_time:.4f} secondi\n")
                file.write("-------------------------------------------------------------\n")

            print(f"VaR: {var_value:.4f}")
            print(f"x: {x_value:.4f}")
            print(f"y1: {y_values[0]:.4f}")
            print(f"y2: {y_values[1]:.4f}")
            print(f"y3: {y_values[2]:.4f}")
            print(f"Tempo di esecuzione: {execution_time:.4f} secondi")
            print("-------------------------------------------------------------")
            
            return var_value
        else:
            print("Ottimizzazione non riuscita.")
            return None
        
        return var_value
    

        
    # Modello per la mimizzazione della varianza con loss semplificata; loss[i] = O[i]   
    def variance_minimize(self, alpha, F0, att, K, Call_prices, dataFutSpot, V, R, num_scenarios, scenarios, log_file):
        model = gp.Model("Minimizzazione_varianza")

        # Variabili
        K1, K2, K3 = K
        C1, C2, C3 = Call_prices
        #lbl = - (max(R))
        # Variabili del modello
        y = model.addVars(3, name="y", lb=0)  
        x = model.addVar(name="x", lb=0) 
        O = model.addVars((scenarios), name="O", lb=0)  
        zw = model.addVars((scenarios), name="zw", lb=-GRB.INFINITY)  
        h = model.addVars(scenarios,3, name="h", lb=0) 
        mean_loss = model.addVar(name="mean_loss", lb=0)
        variance_loss = model.addVar(name="variance_loss", lb=0)
        # Variabili per ciascun termine al quadrato
        squared_differences = model.addVars(scenarios, name="squared_differences", lb=0)

        # Vincoli
        for i in scenarios:
            
            model.addConstr(O[i] == x * F0 + zw[i] * dataFutSpot[i] + att * (C1 * y[0] + C2 * y[1] + C3 * y[2]) + K1 * h[i, 0] + K2 * h[i, 1] + K3 * h[i, 2], name=f"cash_flow_{i}")
            model.addConstr(x + h[i, 0] + h[i, 1] + h[i, 2] + zw[i] == V[i], name=f"balance_{i}")
            model.addConstr(h[i, 0] <= y[0], name=f"h1_constraint_{i}")
            model.addConstr(h[i, 1] <= y[1], name=f"h2_constraint_{i}")
            model.addConstr(h[i, 2] <= y[2], name=f"h3_constraint_{i}")
            
    
        # Media delle perdite
        model.addConstr(
            mean_loss == gp.quicksum(O[i] for i in scenarios) / num_scenarios,
            name="mean_loss_constraint"
        )
        #Vincoli per i termini quadratici della varianza
        model.addConstrs((squared_differences[i] == (O[i] - mean_loss)*(O[i] - mean_loss)  for i in scenarios), name=f"squared_diff_{i}")
    
        # Varianza come norma quadratica
        model.addConstr(
            variance_loss == gp.quicksum(squared_differences[i]  for i in scenarios) / (num_scenarios - 1),
            name="variance_loss_constraint"
        )


        model.setParam('OutputFlag', 0)  
        
        # Ottimizzazione del modello
        start_time = time.time()  
        model.optimize()  
        end_time = time.time()  
        execution_time = end_time - start_time
        
        # Risoluzione
        if model.status == GRB.OPTIMAL:
            variance_value = variance_loss.X
            x_value = x.X
            y_values = [y[i].X for i in range(3)]
            
            # Salvataggio dei dati nel file log
            with open(log_file, mode="a") as file:
                file.write(f"variance: {variance_value:.4f}\n")
                file.write(f"std: {np.sqrt(variance_value):.4f}\n")
                file.write(f"x: {x_value:.4f}\n")
                file.write(f"y1: {y_values[0]:.4f}\n")
                file.write(f"y2: {y_values[1]:.4f}\n")
                file.write(f"y3: {y_values[2]:.4f}\n")
                file.write(f"Tempo di esecuzione: {execution_time:.4f} secondi\n")
                file.write("-------------------------------------------------------------\n")

            print(f"variance: {variance_value:.4f}")
            print(f"std: {np.sqrt(variance_value):.4f}")
            print(f"x: {x_value:.4f}")
            print(f"y1: {y_values[0]:.4f}")
            print(f"y2: {y_values[1]:.4f}")
            print(f"y3: {y_values[2]:.4f}")
            print(f"Tempo di esecuzione: {execution_time:.4f} secondi")
            print("-------------------------------------------------------------")
            return variance_value
        
        else:
            print("Ottimizzazione non riuscita.")
            return None

    # Modello per la mimizzazione dell' EVaR
    def EVaR_minimize(self, alpha, F0, att, K, Call_prices, dataFutSpot, V, R, num_scenarios, scenarios, log_file):
        model = gp.Model("Minimizzazione_varianza")

        # Variabili
        K1, K2, K3 = K
        C1, C2, C3 = Call_prices
        const = - np.log(alpha * num_scenarios) #terminhe costante - logaritmo

        # Variabili del modello
        loss = model.addVars((scenarios), name="loss", lb=0) 
        y = model.addVars(3, name="y", lb=0)  
        x = model.addVar(name="x", lb=0) 
        O = model.addVars((scenarios), name="O", lb=0)  
        zw = model.addVars((scenarios), name="zw", lb=-GRB.INFINITY)  
        h = model.addVars(scenarios,3, name="h", lb=0) 
        EVaR = model.addVar(name="EVaR", lb=0)
        z = model.addVar(name="z", lb=0)
        z_inv = model.addVar(name="z_inv")
        prod_terms = model.addVars(scenarios, name="prod_terms", lb=0)  # z * loss[i]
        gen_momenti = model.addVars(scenarios, name ="gen_mom",lb=0)
        sum_exp = model.addVar(name="sum_exp", lb=0)                 # Somma degli e^(z * loss[i])
        log_sum_exp = model.addVar(name="log_sum_exp", lb=0 ) # ln(sum_exp / n)
        
        
        for i in scenarios:
            # Vincolo per z
            model.addConstr(O[i] == x * F0 + zw[i] * dataFutSpot[i] + att * (C1 * y[0] + C2 * y[1] + C3 * y[2]) + K1 * h[i,0] + K2 * h[i,1] + K3 * h[i,2], name=f"bilancio flusso di cassa in uscita scenario {i}" )
            model.addConstr(x + h[i, 0] + h[i, 1] + h[i, 2] + zw[i] == V[i], name=f"bilancio volume in uscita scenario {i}")
            model.addConstr(h[i, 0] <= y[0], name=f"h1_constraint_{i}")
            model.addConstr(h[i, 1] <= y[1], name=f"h2_constraint_{i}")
            model.addConstr(h[i, 2] <= y[2], name=f"h3_constraint_{i}")
            # Vincolo per legare loss[i] al massimo tra 0 e O[i] - R_s[i]
            model.addConstrs(loss[i] >= O[i] - R[i]  for i in scenarios)  # Deve essere almeno O[i] - R_s[i]
            model.addConstr(prod_terms[i] == z * loss[i], name=f"prod_calc_{i}")
            # Calcolo di e^(z * loss[i]) per ciascun scenario
            model.addGenConstrExp(prod_terms[i], gen_momenti[i], name=f"exp_calc_{i}")
        
         
        model.addConstr(sum_exp == gp.quicksum(gen_momenti[i] for i in scenarios))
        # Calcolo del logaritmo del valore medio degli esponenziali
        model.addGenConstrLog(sum_exp, log_sum_exp, name="log_calc")
        
        model.addConstr(z * z_inv == 1, name="z_inverse_constraint")
        
        # Vincolo per l'EVaR
        model.addConstr(EVaR == z_inv * (const + log_sum_exp), name="EVaR_definition")

        # Funzione obiettivo
        model.setObjective(EVaR, GRB.MINIMIZE)
        # Attiva il log di ottimizzazione
        model.setParam('OutputFlag', 0)  # Imposta a 1 per il log di ottimizzazione
        
        # Ottimizzazione del modello
        start_time = time.time()  
        model.optimize()  
        end_time = time.time()  
        execution_time = end_time - start_time
        
        # Risoluzione
        if model.status == GRB.OPTIMAL:
            EVaR_value = EVaR.X
            x_value = x.X
            y_values = [y[i].X for i in range(3)]
            
            # Salvataggio dei dati nel file log
            with open(log_file, mode="a") as file:
                file.write(f"EVaR: {EVaR_value:.4f}\n")
                file.write(f"x: {x_value:.4f}\n")
                file.write(f"y1: {y_values[0]:.4f}\n")
                file.write(f"y2: {y_values[1]:.4f}\n")
                file.write(f"y3: {y_values[2]:.4f}\n")
                file.write(f"Tempo di esecuzione: {execution_time:.4f} secondi\n")
                file.write("-------------------------------------------------------------\n")

            print(f"EVaR: {EVaR_value:.4f}")
            print(f"x: {x_value:.4f}")
            print(f"y1: {y_values[0]:.4f}")
            print(f"y2: {y_values[1]:.4f}")
            print(f"y3: {y_values[2]:.4f}")
            print(f"Tempo di esecuzione: {execution_time:.4f} secondi")
            print("-------------------------------------------------------------")
            return EVaR_value
        
        else:
            print("Ottimizzazione non riuscita.")
            return None
        
        
    
