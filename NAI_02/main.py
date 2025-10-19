"""
Smart Charger - Proof of Concept z logiką rozmytą

Autorzy: Kacper Olejnik, Hubert Jóźwiak

Wejścia: SoC (0-100%), Temperatura (0-60°C), Wiek baterii (0-1000 cykli)
Wyjście: Prąd ładowania (0-3A)
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Definicja zmiennych wejściowych i wyjściowych
soc = ctrl.Antecedent(np.arange(0, 101, 1), 'soc')
temp = ctrl.Antecedent(np.arange(0, 61, 1), 'temp')
age = ctrl.Antecedent(np.arange(0, 1001, 10), 'age')
current = ctrl.Consequent(np.arange(0, 3.1, 0.1), 'current')

# Funkcje przynależności dla SoC
soc['niski'] = fuzz.trapmf(soc.universe, [0, 0, 20, 40])
soc['sredni'] = fuzz.trimf(soc.universe, [30, 55, 80])
soc['wysoki'] = fuzz.trapmf(soc.universe, [70, 90, 100, 100])

# Funkcje przynależności dla temperatury
temp['zimna'] = fuzz.trapmf(temp.universe, [0, 0, 10, 20])
temp['optymalna'] = fuzz.trapmf(temp.universe, [15, 20, 30, 35])
temp['goraca'] = fuzz.trapmf(temp.universe, [30, 45, 60, 60])

# Funkcje przynależności dla wieku
age['nowa'] = fuzz.trapmf(age.universe, [0, 0, 200, 400])
age['stara'] = fuzz.trapmf(age.universe, [300, 700, 1000, 1000])

# Funkcje przynależności dla prądu
current['wolne'] = fuzz.trimf(current.universe, [0, 0.5, 1.2])
current['normalne'] = fuzz.trimf(current.universe, [1.0, 1.8, 2.5])
current['szybkie'] = fuzz.trimf(current.universe, [2.0, 2.5, 3.0])

# Reguły rozmyte
rule1 = ctrl.Rule(temp['goraca'], current['wolne'])
rule2 = ctrl.Rule(soc['wysoki'], current['wolne'])
rule3 = ctrl.Rule(soc['niski'] & temp['optymalna'] & age['nowa'], current['szybkie'])
rule4 = ctrl.Rule(soc['niski'] & temp['optymalna'] & age['stara'], current['normalne'])
rule5 = ctrl.Rule(soc['sredni'] & temp['optymalna'], current['normalne'])
rule6 = ctrl.Rule(age['stara'], current['wolne'])
rule7 = ctrl.Rule(temp['zimna'], current['wolne'])

# Utworzenie systemu sterowania
charging_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
charging = ctrl.ControlSystemSimulation(charging_ctrl)


# Funkcja testująca
def test_charger(soc_val, temp_val, age_val):
    charging.input['soc'] = soc_val
    charging.input['temp'] = temp_val
    charging.input['age'] = age_val
    charging.compute()
    return charging.output['current']


# Symulacja ładowania
print("Smart Charger - Symulacja Ładowania\n")
print("Krok | SoC  | Temp | Wiek | Prąd  | Status")
print("-" * 50)

# Parametry symulacji
soc_val = 0.0
temp_val = 22.0
age_val = 300  # Bateria średniego wieku
step = 0

# Symulacja ładowania do 100%
while soc_val < 100 and step < 30:
    step += 1
    
    # Oblicz prąd ładowania
    curr = test_charger(soc_val, temp_val, age_val)
    
    # Określ status
    if curr >= 2.0:
        status = "Szybkie"
    elif curr >= 1.2:
        status = "Normalne"
    else:
        status = "Wolne"
    
    # Wyświetl aktualny stan
    print(f"{step:4d} | {soc_val:3.0f}% | {temp_val:2.0f}°C | {age_val:4d} | {curr:.2f}A | {status}")
    
    # Symuluj wzrost SoC (im większy prąd, tym szybszy wzrost)
    soc_val += curr * 3.0
    
    # Symuluj wzrost temperatury (ładowanie generuje ciepło)
    temp_val += curr * 0.3
    
    # Ogranicz wartości
    if soc_val >= 100:
        soc_val = 100
    if temp_val > 60:
        temp_val = 60

print(f"\nŁadowanie zakończone! Bateria naładowana do {soc_val:.0f}%")