"""
Klasyfikacja danych (Banknote, Diabetes) przy użyciu Drzewa Decyzyjnego i SVM.
Analiza wpływu funkcji jądra (Kernels) oraz wizualizacja wyników.

Autorzy: Hubert Jóźwiak, Kacper Olejnik
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. KONFIGURACJA I POBIERANIE DANYCH
# ============================================================

# Zbiór 1: Banknote Authentication (zgodnie z poleceniem)
url_banknote = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
cols_banknote = ["variance", "skewness", "kurtosis", "entropy", "class"]

# Zbiór 2: Pima Indians Diabetes (wybrany unikatowy zbiór)
url_diabetes = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols_diabetes = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]

def load_data(url, columns):
    """Pobiera dane z URL i dzieli na cechy (X) oraz etykiety (y)."""
    df = pd.read_csv(url, names=columns)
    X = df.drop('class', axis=1)
    y = df['class']
    return df, X, y

# ============================================================
# 2. FUNKCJE POMOCNICZE (WIZUALIZACJA I TRENING)
# ============================================================

def visualize_data(df, title):
    """
    Tworzy wykres Pairplot (macierz wykresów rozrzutu).
    Pozwala ocenić wizualnie, czy klasy są łatwo separowalne.
    """
    print(f"Generowanie wykresu dla: {title}...")
    sns.pairplot(df, hue='class', markers=["o", "s"], diag_kind="kde")
    plt.suptitle(f'Wizualizacja danych: {title}', y=1.02)
    plt.show()

def train_evaluate_classifiers(X, y, dataset_name):
    """
    Trenuje Drzewo Decyzyjne i SVM, a następnie wyświetla metryki.
    Zwraca wytrenowane modele i skaler do późniejszego użycia.
    """
    print(f"\n{'='*20} ANALIZA ZBIORU: {dataset_name} {'='*20}")
    
    # Krok 1: Podział na zbiór treningowy i testowy (70% trening, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Krok 2: Skalowanie danych (Krytyczne dla SVM!)
    # Drzewa nie wymagają skalowania, ale SVM działa znacznie gorzej na surowych danych.
    scaler = StandardScaler()
    # Fitujemy tylko na treningowym, transformujemy oba. Zwraca DataFrame (dzięki set_output), żeby zachować nazwy kolumn.
    scaler.set_output(transform="pandas") 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- A. KLASYFIKATOR: DRZEWO DECYZYJNE ---
    # Używamy entropii (zgodnie z sugestią w artykule ResearchGate)
    dt = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    print(f"\n>>> Raport klasyfikacji: Drzewo Decyzyjne ({dataset_name})")
    print(classification_report(y_test, y_pred_dt))
    
    # --- B. KLASYFIKATOR: SVM (Support Vector Machine) ---
    # Domyślny kernel RBF
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    
    print(f"\n>>> Raport klasyfikacji: SVM RBF ({dataset_name})")
    print(classification_report(y_test, y_pred_svm))

    return dt, svm, scaler

def compare_svm_kernels(X, y):
    """
    Prezentacja użycia różnych rodzajów kernel function z różnymi parametrami.
    Pokazuje wpływ jądra na dokładność (Accuracy).
    """
    print(f"\n--- PORÓWNANIE KERNELI SVM (Dla zbioru danych) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Lista różnych konfiguracji kerneli
    kernels_params = [
        ('linear', {}),                 # Kernel liniowy
        ('poly', {'degree': 3}),        # Wielomianowy stopnia 3
        ('rbf', {'gamma': 'scale'}),    # Radial Basis Function (standard)
        ('sigmoid', {})                 # Sigmoidalny (często trudny w treningu)
    ]
    
    for name, params in kernels_params:
        clf = SVC(kernel=name, **params, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Kernel: {name:10} | Parametry: {str(params):15} | Accuracy: {acc:.4f}")

# ============================================================
# 3. GŁÓWNA LOGIKA PROGRAMU (WYKONANIE)
# ============================================================

# --- ZBIÓR 1: BANKNOTE ---
df_bn, X_bn, y_bn = load_data(url_banknote, cols_banknote)
visualize_data(df_bn, "Banknote Authentication") # Wizualizacja

# Trening i testy dla Banknote
dt_model, svm_model, scaler_model = train_evaluate_classifiers(X_bn, y_bn, "BANKNOTE")

# Analiza wpływu Kerneli (dla Banknote)
compare_svm_kernels(X_bn, y_bn)


# --- ZBIÓR 2: DIABETES ---
df_diab, X_diab, y_diab = load_data(url_diabetes, cols_diabetes)
# visualize_data(df_diab, "Pima Diabetes") # Opcjonalnie

# Trening i testy dla Diabetes
train_evaluate_classifiers(X_diab, y_diab, "DIABETES")


# ============================================================
# 4. WYWOŁANIE KLASYFIKATORÓW DLA PRZYKŁADOWYCH DANYCH
# ============================================================
print(f"\n{'='*20} DEMO: WYWOŁANIE (INFERENCJA) {'='*20}")

# Krok A: Definicja przykładowych danych wejściowych (symulacja nowego odczytu)
# Wartości odpowiadają kolumnom: [variance, skewness, kurtosis, entropy]
raw_input_data = [[0.5, -1.2, 3.0, -0.5]]

# Krok B: Konwersja do DataFrame z nazwami kolumn
# To kluczowy moment, aby uniknąć warningu "X does not have valid feature names"
input_df = pd.DataFrame(raw_input_data, columns=cols_banknote[:-1])

print("Dane wejściowe (Nowy banknot):")
print(input_df.to_string(index=False))

# Krok C: Wywołanie Drzewa Decyzyjnego
# Drzewo przyjmuje dane w takiej formie, w jakiej trenowano (DataFrame z nazwami)
prediction_dt = dt_model.predict(input_df)[0]

# Krok D: Wywołanie SVM
# SVM wymagał skalowania podczas treningu, więc nowe dane też musimy przeskalować
# Używamy tego samego obiektu 'scaler_model', który "nauczył się" średniej i odchylenia na etapie treningu
input_scaled = scaler_model.transform(input_df)
prediction_svm = svm_model.predict(input_scaled)[0]

# Krok E: Prezentacja wyników
print("\n--- WYNIKI PREDYKCJI ---")
print(f"Drzewo Decyzyjne -> Klasa: {prediction_dt} ({'Fałszywy' if prediction_dt==1 else 'Prawdziwy'})")
print(f"SVM (RBF)        -> Klasa: {prediction_svm} ({'Fałszywy' if prediction_svm==1 else 'Prawdziwy'})")