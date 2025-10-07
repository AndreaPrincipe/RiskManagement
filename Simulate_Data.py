import numpy as np
import random
from scipy.stats import norm

class Simulate_Data:
    
    def __init__(self, array_students, distr,S0, rf, rd, volatility, T, num_scenarios):
        
        self.num_scenarios = num_scenarios #num volte che li voglio simulare
        
        # Dati per gli studenti
        self.array_students = array_students
        self.distr = distr
        
        
        # Dati rimanenti
        self.S0 = S0
        self.rf = rf
        self.rd = rd
        self.volatility = volatility
        self.T = T
        
    # Metodo che simula gli studenti partecipanti    
    def simulate_students(self):
        # Distribuzione : uniforme
        if self.distr  == 'uniform':
            probs_stud = np.ones(len(self.array_students)) / len(self.array_students);
            probs_stud /= probs_stud.sum()
            simulated_students = np.random.choice(self.array_students, size = self.num_scenarios, p = probs_stud)
         
        # Distribuzione: uniforme lineare a tratti, simmetrica
        elif self.distr  == 'pw_uniform':
            
            # Numero totale di studenti
            num_students = len(self.array_students)  
            start = self.array_students[0];
            end = self.array_students[-1];
            # Numero di segmenti per la distribuzione a gradini
            num_segments = 100  

            # Suddivisione dell'intervallo in segmenti
            segment_edges = np.linspace(start, end, num_segments + 1)

            # Probabilità a gradini: crescente fino a metà, poi decrescente
            probabilities = np.concatenate((
                np.linspace(0.1, 1.0, num_segments // 2),  
                np.linspace(1.0, 0.1, num_segments // 2)  
            ))

            # Normalizza le probabilità dei segmenti
            probabilities /= probabilities.sum()

            # Assegna le probabilità agli studenti in base ai segmenti
            probs_stud = np.zeros(num_students)

            for i in range(num_segments):
                # Studenti in ogni segmento
                mask = (self.array_students >= segment_edges[i]) & (self.array_students < segment_edges[i + 1])
                # Per gli studenti in quel segmento, al probabilità è:
                probs_stud[mask] = probabilities[i]


            probs_stud /= probs_stud.sum()
            
            # Numero studenti generato per ogni scenario
            simulated_students = np.random.choice(self.array_students, size = self.num_scenarios, p = probs_stud)
         
        #Distribuzione: Beta   
        elif self.distr  == 'Beta':  
            
             # Parametri della distribuzione Beta
             alpha = 3  
             beta = 4

             # Generazione campioni dalla  Beta
             samples_beta = np.random.beta(alpha, beta, self.num_scenarios)

             # Riscalati nel range di studenti
             simulated_students = self.array_students[0] + samples_beta * (self.array_students[-1] - self.array_students[0])
             # Arrotondiamo all'intero più vicino
             simulated_students = np.round(simulated_students).astype(int)
            
        return simulated_students
    
    # Calcolo prezzo spot futuri (GBM, misura neutrale al rischio)
    def simulate_fut_Spot(self):
        Z = np.random.normal(0, 1, self.num_scenarios)
        X = (self.rd - self.rf - (self.volatility** 2) / 2) * self.T + self.volatility* np.sqrt(self.T) * Z
        # Calcolo prezzi spot: è un  vettore contenente num_scenarios valori simulati del prezzo spot futuro in T
        ST = self.S0 * np.exp(X) 
        
        return ST
    
