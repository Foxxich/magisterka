import os
import sys
import time
from common import evaluate_model, get_bert_embeddings, get_roberta_embeddings, load_and_preprocess_data, split_data, split_data_few_shot, split_data_one_shot, get_transformer_embeddings

# Ścieżki do katalogów z metodami
for i in range(1, 17):
    sys.path.append(os.path.join(os.getcwd(), f'article{i}'))

# Import funkcji treningowych
from run1 import metoda1
from run2 import metoda2
from run3 import metoda3
from run4 import metoda4
from run5 import metoda5
from run6 import metoda6
from run7 import metoda7
from run8 import metoda8
from run9 import metoda9
from run10 import metoda10
from run11 import metoda11
from run12 import metoda12
from run13 import metoda13
from run14 import metoda14
from run15 import metoda15
from run16 import metoda16
from my_run import metoda17
from my_run import metoda18
from my_run import metoda19
from my_run import metoda20

# Ładowanie zmiennych środowiskowych
from dotenv import load_dotenv
load_dotenv()

dataset_input = ""

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
        models, X_test, y_test = metoda1(X_train, y_train, X_test, y_test)
        for i, model in enumerate(models):
            run_name = f"run1-{i + 1}"  # Unikalna nazwa dla każdego modelu
            evaluate_model(model, X_test, y_test, run_name, output_path, start_time, dataset_input)
    elif method_number == 2:
        model, X_test, y_test = metoda2(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run2", output_path, start_time, dataset_input)
    elif method_number == 3:
        model, X_test, y_test = metoda3(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run3", output_path, start_time, dataset_input)
    elif method_number == 4:
        model, X_test, y_test = metoda4(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run4", output_path, start_time, dataset_input)
    elif method_number == 5:
        model, meta_test, y_test = metoda5(X_train, y_train, X_test, y_test)
        evaluate_model(model, meta_test, y_test, "run5", output_path, start_time, dataset_input)
    elif method_number == 6:
        model, meta_test, y_test = metoda6(X_train, y_train, X_test, y_test)
        evaluate_model(model, meta_test, y_test, "run6", output_path, start_time, dataset_input)
    elif method_number == 7:
        model, X_test, y_test = metoda7(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run7", output_path, start_time, dataset_input)
    elif method_number == 8:
        model, X_test, y_test = metoda8(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run8", output_path, False)
    elif method_number == 9:
        model, X_test, y_test = metoda9(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run9", output_path, start_time, dataset_input)
    elif method_number == 10:
        model, X_test, y_test = metoda10(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run10", output_path, start_time, dataset_input)
    elif method_number == 11:
        model, X_test, y_test = metoda11(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run11", output_path, start_time, dataset_input)
    elif method_number == 12:
        models = metoda12(X_train, y_train, X_test, y_test)
        evaluate_model(models["RandomForest"]["model"], X_test, y_test, "run12-rf", output_path, start_time, dataset_input)
        evaluate_model(models["CatBoost"]["model"], X_test, y_test, "run12-catboost", output_path, start_time, dataset_input)
    elif method_number == 13:
        model, X_test, y_test = metoda13(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run13", output_path, start_time, dataset_input)
    elif method_number == 14:
        model, X_test, y_test = metoda14(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run14", output_path, start_time, dataset_input)
    elif method_number == 15:
        model, X_test, y_test = metoda15(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run15", output_path, start_time, dataset_input)
    elif method_number == 16:
        model, X_test, y_test = metoda16(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run16", output_path, start_time, dataset_input)
    elif method_number == 17:
        model, X_test, y_test = metoda17(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run17", output_path, start_time, dataset_input)
    elif method_number == 18:
        model, X_test, y_test = metoda18(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run18", output_path, start_time, dataset_input)
    elif method_number == 19:
        model, X_test, y_test = metoda19(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run19", output_path, start_time, dataset_input)
    elif method_number == 20:
        model, X_test, y_test = metoda20(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test, "run20", output_path, start_time, dataset_input)
    else:
        print(f"Metoda {method_number} nie istnieje!")

if __name__ == "__main__":
    print("Wybierz reprezentację kontekstową (1-3) lub wpisz 'all', aby uruchomić wszystkie:\n1 - BERT\n2 - RoBERTa\n3 - Sentence Transformers")
    representation_input = input().strip().lower()

    if representation_input == "all":
        print("Uruchamiam wszystkie reprezentacje...")
        print("Wybierz numer metody początkującej (1-20) lub wpisz 'all', aby uruchomić wszystkie:")
        method_input_first = input().strip().lower()
        method_number_first = 1
        method_number_last = 20
        if method_input_first != "all":
            method_number_first = int(method_input_first)
            print("Wybierz numer metody ostatecznej (1-20)")
            method_input_last = input().strip().lower()
            method_number_last = int(method_input_last) + 1

        for rep in ["3"]:
            print(f"Rozpoczynanie metod dla reprezentacji {rep}...")
            for split_type in ["classic", "one_shot", "few_shot"]:
                for dataset_input in ["ISOT", "BuzzFeed", "WELFake"]:
                    X, y = load_and_preprocess_data(dataset_input)
                    X_embeddings = get_embeddings(rep, X, y)
                    for method_number in range(method_number_first, method_number_last):
                        print(f"Uruchamianie metody {method_number} dla reprezentacji {rep} i trybu {split_type} oraz danych {dataset_input}...")
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
        print("Wybierz który dataset zostanie użyty:\n1 - ISOT\n2 - BuzzFeed\n3 - WELFake")     
        dataset_input_user = input().strip().lower()
        print("Wybierz numer metody początkującej (1-20) lub wpisz 'all', aby uruchomić wszystkie:")
        method_input_first = input().strip().lower()
        method_number_first = 1
        method_number_last = 20
        if method_input_first != "all":
            method_number_first = int(method_input_first)
            print("Wybierz numer metody ostatecznej (1-20)")
            method_input_last = input().strip().lower()
            method_number_last = int(method_input_last) + 1

        print("Wybierz tryb podziału danych (1-3) lub wpisz 'all', aby uruchomić wszystkie:\n1 - Klasyczny split\n2 - One Shot\n3 - Few Shot")
        split_input = input().strip().lower()
        print(f"Generowanie reprezentacji kontekstowej {representation_input}...")
        if dataset_input_user == 1:
            dataset_input = "ISOT"
        elif dataset_input_user == 2:
            dataset_input = "BuzzFeed"
        else:
            dataset_input = "WELFake"   
        X, y = load_and_preprocess_data(dataset_input)
        X_embeddings = get_embeddings(representation_input, X, y)
        if X_embeddings is not None:
            if split_input == "all":
                for split_type in ["classic", "one_shot", "few_shot"]:
                    if method_number_first == "all":
                        for method_number in range(1, method_number_last + 1):
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
                        for method_number in range(method_number_first, method_number_last):
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

                if method_input_first == "all":
                    for method_number in range(1, method_number_last + 1):
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
                    for method_number in range(method_number_first, method_number_last):
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