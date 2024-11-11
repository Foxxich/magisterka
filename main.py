import os
import sys
from common import evaluate_model

# Ścieżki do katalogów z metodami
for i in range(1, 17):
    sys.path.append(os.path.join(os.getcwd(), f'article{i}'))

# Import funkcji treningowych
from run1 import train_run1
from run2 import train_run2
from run3 import train_run3
from run4 import train_run4
from run5 import train_run5
from run6 import train_run6
from run7 import train_run7
from run8 import train_run8
from run9 import train_run9
from run10 import train_run10
from run11 import train_run11
from run12 import train_run12
from run13 import train_run13
from run14 import train_run14
from run15 import train_run15
from run16 import train_run16

# Ścieżka do zapisu wyników
OUTPUT_PATH = r"C:\Users\Vadym\Documents\magisterka\results"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def run_method(method_number):
    """Uruchamia wybraną metodę na podstawie numeru."""
    if method_number == 1:
        model, X_test, y_test = train_run1()
        evaluate_model(model, X_test, y_test, "run1", OUTPUT_PATH)
    elif method_number == 2:
        model, X_test, y_test = train_run2()
        evaluate_model(model, X_test, y_test, "run2", OUTPUT_PATH)
    elif method_number == 3:
        model, X_test, y_test = train_run3()
        evaluate_model(model, X_test, y_test, "run3", OUTPUT_PATH)
    elif method_number == 4:
        model, X_test, y_test = train_run4()
        evaluate_model(model, X_test, y_test, "run4", OUTPUT_PATH)
    elif method_number == 5:
        model, meta_test, y_test = train_run5()
        evaluate_model(model, meta_test, y_test, "run5", OUTPUT_PATH)
    elif method_number == 6:
        model, meta_test, y_test = train_run6()
        evaluate_model(model, meta_test, y_test, "run6", OUTPUT_PATH)
    elif method_number == 7:
        model, X_test, y_test, final_preds = train_run7()
        evaluate_model(model, X_test, y_test, "run7", OUTPUT_PATH)
    elif method_number == 8:
        model, X_test, y_test = train_run8()
        evaluate_model(model, X_test, y_test, "run8", OUTPUT_PATH, False)
    elif method_number == 9:
        model, X_test, y_test = train_run9()
        evaluate_model(model, X_test, y_test, "run9", OUTPUT_PATH)
    elif method_number == 10:
        model, X_test, y_test = train_run10()
        evaluate_model(model, X_test, y_test, "run10", OUTPUT_PATH)
    elif method_number == 11:
        model, X_test, y_test = train_run11()
        evaluate_model(model, X_test, y_test, "run11", OUTPUT_PATH)
    elif method_number == 12:
        models = train_run12()
        evaluate_model(models["RandomForest"][0], models["RandomForest"][1], models["RandomForest"][2], "run12_rf", OUTPUT_PATH)
        evaluate_model(models["CatBoost"][0], models["CatBoost"][1], models["CatBoost"][2], "run12_catboost", OUTPUT_PATH)
    elif method_number == 13:
        model, X_test, y_test = train_run13()
        evaluate_model(model, X_test, y_test, "run13", OUTPUT_PATH)
    elif method_number == 14:
        model, X_test, y_test = train_run14()
        evaluate_model(model, X_test, y_test, "run14", OUTPUT_PATH)
    elif method_number == 15:
        model, X_test, y_test = train_run15()
        evaluate_model(model, X_test, y_test, "run15", OUTPUT_PATH)
    elif method_number == 16:
        model, X_test, y_test = train_run16()
        evaluate_model(model, X_test, y_test, "run16", OUTPUT_PATH)

    else:
        print(f"Metoda {method_number} nie istnieje!")

def run_all_methods():
    """Uruchamia wszystkie metody po kolei."""
    for i in range(1, 17):
        print(f"Uruchamianie metody {i}...")
        run_method(i)

if __name__ == "__main__":
    print("Podaj numer metody do uruchomienia (1-16) lub wpisz 'all', aby uruchomić wszystkie:")
    user_input = input().strip().lower()
    if user_input == "all":
        run_all_methods()
    else:
        try:
            method_number = int(user_input)
            run_method(method_number)
        except ValueError:
            print("Nieprawidłowy wybór. Podaj liczbę od 1 do 16 lub wpisz 'all'.")
