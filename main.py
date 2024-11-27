import os
import sys
import time
from common import evaluate_model, get_bert_embeddings, get_roberta_embeddings, load_and_preprocess_data, split_data, split_data_few_shot, split_data_one_shot, vectorize_data, get_transformer_embeddings

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

# Ładowanie zmiennych środowiskowych
from dotenv import load_dotenv
load_dotenv()

def get_embeddings(option, X, y):
    """
    Generuje reprezentację kontekstową na podstawie wybranej opcji.
    """
    if option == "1":
        return get_bert_embeddings(X.tolist())
    elif option == "2":
        return get_roberta_embeddings(X.tolist())
    elif option == "3":
        return get_transformer_embeddings(X.tolist())
    else:
        print(f"Nieprawidłowa opcja reprezentacji kontekstowej: {option}.")
        return None

def run_method(method_number, X_train, y_train, X_test, y_test, output_path):
    """Uruchamia wybraną metodę na podstawie numeru."""
    start_time = time.time()
    if method_number == 1:
        model, X_test, y_test = train_run1(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run1", output_path, start_time)
    elif method_number == 2:
        model, X_test, y_test = train_run2(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run2", output_path, start_time)
    elif method_number == 3:
        model, X_test, y_test = train_run3(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run3", output_path, start_time)
    elif method_number == 4:
        model, X_test, y_test = train_run4(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run4", output_path, start_time)
    elif method_number == 5:
        model, meta_test, y_test = train_run5(X_train, y_train, X_test, y_test)
        evaluate_model(model, meta_test, y_test, "run5", output_path, start_time)
    elif method_number == 6:
        model, meta_test, y_test = train_run6(X_train, y_train, X_test, y_test)
        evaluate_model(model, meta_test, y_test, "run6", output_path, start_time)
    elif method_number == 7:
        model, X_test, y_test = train_run7(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run7", output_path, start_time)
    elif method_number == 8:
        model, X_test, y_test = train_run8(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run8", output_path, False)
    elif method_number == 9:
        model, X_test, y_test = train_run9(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run9", output_path, start_time)
    elif method_number == 10:
        model, X_test, y_test = train_run10(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run10", output_path, start_time)
    elif method_number == 11:
        model, X_test, y_test = train_run11(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run11", output_path, start_time)
    elif method_number == 12:
        models = train_run12(X_train, y_train, X_test, y_test)
        evaluate_model(models["RandomForest"][0], models["RandomForest"][1], models["RandomForest"][2], "run12_rf", output_path, start_time)
        evaluate_model(models["CatBoost"][0], models["CatBoost"][1], models["CatBoost"][2], "run12_catboost", output_path, start_time)
    elif method_number == 13:
        model, X_test, y_test = train_run13(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run13", output_path, start_time)
    elif method_number == 14:
        model, X_test, y_test = train_run14(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run14", output_path, start_time)
    elif method_number == 15:
        model, X_test, y_test = train_run15(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run15", output_path, start_time)
    elif method_number == 16:
        model, X_test, y_test = train_run16(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run16", output_path, start_time)
    else:
        print(f"Metoda {method_number} nie istnieje!")

if __name__ == "__main__":
    print("Wybierz reprezentację kontekstową (1-3) lub wpisz 'all', aby uruchomić wszystkie:\n1 - BERT\n2 - RoBERTa\n3 - Sentence Transformers")
    representation_input = input().strip().lower()

    X, y = load_and_preprocess_data()

    if representation_input == "all":
        print("Uruchamiam wszystkie reprezentacje...")
        for rep in ["1"]:
            print(f"Generowanie reprezentacji kontekstowej {rep}...")
            X_embeddings = get_embeddings(rep, X, y)
            if X_embeddings is not None:
                print(f"Rozpoczynanie metod dla reprezentacji {rep}...")
                for split_type in ["classic", "one_shot", "few_shot"]:
                    for method_number in range(1,17):
                        print(f"Uruchamianie metody {method_number} dla reprezentacji {rep} i trybu {split_type}...")
                        if split_type == "classic":
                            X_train, X_test, y_train, y_test = split_data(X_embeddings, y)
                        elif split_type == "one_shot":
                            X_train, X_test, y_train, y_test = split_data_one_shot(X_embeddings, y)
                        elif split_type == "few_shot":
                            X_train, X_test, y_train, y_test = split_data_few_shot(X_embeddings, y)
                        try:
                            run_method(method_number, X_train, y_train, X_test, y_test, f"results_{rep}_{split_type}")
                        except Exception as e:
                            print(f"Wystąpił błąd podczas uruchamiania metody {method_number} dla trybu {split_type}: {e}")

    else:
        print(f"Generowanie reprezentacji kontekstowej {representation_input}...")
        X_embeddings = get_embeddings(representation_input, X, y)
        if X_embeddings is not None:
            print("Wybierz numer metody (1-16) lub wpisz 'all', aby uruchomić wszystkie:")
            method_input = input().strip().lower()

            print("Wybierz tryb podziału danych (1-3) lub wpisz 'all', aby uruchomić wszystkie:\n1 - Klasyczny split\n2 - One Shot\n3 - Few Shot")
            split_input = input().strip().lower()

            if split_input == "all":
                for split_type in ["classic", "one_shot", "few_shot"]:
                    if method_input == "all":
                        for method_number in range(1, 17):
                            print(f"Uruchamianie metody {method_number} dla reprezentacji {representation_input} i trybu {split_type}...")
                            if split_type == "classic":
                                X_train, X_test, y_train, y_test = split_data(X_embeddings, y)
                            elif split_type == "one_shot":
                                X_train, X_test, y_train, y_test = split_data_one_shot(X_embeddings, y)
                            elif split_type == "few_shot":
                                X_train, X_test, y_train, y_test = split_data_few_shot(X_embeddings, y)
                            try:
                                run_method(method_number, X_train, y_train, X_test, y_test, f"results_{representation_input}_{split_type}")
                            except Exception as e:
                                print(f"Wystąpił błąd podczas uruchamiania metody {method_number} dla trybu {split_type}: {e}")

                    else:
                        method_number = int(method_input)
                        print(f"Uruchamianie metody {method_number} dla reprezentacji {representation_input} i trybu {split_type}...")
                        if split_type == "classic":
                            X_train, X_test, y_train, y_test = split_data(X_embeddings, y)
                        elif split_type == "one_shot":
                            X_train, X_test, y_train, y_test = split_data_one_shot(X_embeddings, y)
                        elif split_type == "few_shot":
                            X_train, X_test, y_train, y_test = split_data_few_shot(X_embeddings, y)
                        try:
                            run_method(method_number, X_train, y_train, X_test, y_test, f"results_{representation_input}_{split_type}")
                        except Exception as e:
                            print(f"Wystąpił błąd podczas uruchamiania metody {method_number} dla trybu {split_type}: {e}")

            else:
                if split_input == "1":
                    split_type = "classic"
                elif split_input == "2":
                    split_type = "one_shot"
                elif split_input == "3":
                    split_type = "few_shot"
                else:
                    print("Nieprawidłowy wybór trybu. Domyślnie używam klasycznego splitu.")
                    split_type = "classic"

                if method_input == "all":
                    for method_number in range(1, 17):
                        print(f"Uruchamianie metody {method_number} dla reprezentacji {representation_input} i trybu {split_type}...")
                        if split_type == "classic":
                            X_train, X_test, y_train, y_test = split_data(X_embeddings, y)
                        elif split_type == "one_shot":
                            X_train, X_test, y_train, y_test = split_data_one_shot(X_embeddings, y)
                        elif split_type == "few_shot":
                            X_train, X_test, y_train, y_test = split_data_few_shot(X_embeddings, y)
                        try:
                            run_method(method_number, X_train, y_train, X_test, y_test, f"results_{representation_input}_{split_type}")
                        except Exception as e:
                            print(f"Wystąpił błąd podczas uruchamiania metody {method_number} dla trybu {split_type}: {e}")

                else:
                    method_number = int(method_input)
                    print(f"Uruchamianie metody {method_number} dla reprezentacji {representation_input} i trybu {split_type}...")
                    if split_type == "classic":
                        X_train, X_test, y_train, y_test = split_data(X_embeddings, y)
                    elif split_type == "one_shot":
                        X_train, X_test, y_train, y_test = split_data_one_shot(X_embeddings, y)
                    elif split_type == "few_shot":
                        X_train, X_test, y_train, y_test = split_data_few_shot(X_embeddings, y)
                    try:
                        run_method(method_number, X_train, y_train, X_test, y_test, f"results_{representation_input}_{split_type}")
                    except Exception as e:
                        print(f"Wystąpił błąd podczas uruchamiania metody {method_number} dla trybu {split_type}: {e}")