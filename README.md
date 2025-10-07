# 🎯 Risk Measure Optimization and Simulation (CVaR, VaR, EVaR, Variance)



## 📘 Overview

This project implements a **Monte Carlo simulation** framework for evaluating and optimizing different **financial risk measures**, specifically:

- **CVaR** (Conditional Value at Risk)  
- **VaR** (Value at Risk)  
- **EVaR** (Entropic Value at Risk)  
- **Variance**

The model simulates a foreign exchange ($/€) exposure related to **student funding programs**, and optimizes hedging strategies using **call options** on the exchange rate.

The goal is to **minimize the chosen risk measure** by simulating random scenarios of:
- the number of students,
- future exchange rates,  
and computing the total cost in euros under each scenario.

---

## 🧭 Project Structure

project_root/
│
├── main.py # Main simulation script
├── Simulate_Data.py # Module for generating simulated data (students, FX rates)
├── risk_models.py # Module containing risk measure optimization models
├── simulation_log_CVaR.txt # Example log file (auto-generated)
│

