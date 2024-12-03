
# Praca Magisterska

Projekt dotyczy wykrywania fałszywych wiadomości przy użyciu zaawansowanych metod uczenia maszynowego. Zawiera moduły do wstępnego przetwarzania danych, ekstrakcji cech, trenowania oraz oceny modeli w różnych konfiguracjach.

## Spis treści

- [Wprowadzenie](#wprowadzenie)
- [Wymagania](#wymagania)
- [Instalacja](#instalacja)
- [Uruchamianie](#uruchamianie)
- [Pliki Projektu](#pliki-projektu)

---

## Wprowadzenie

Projekt ma na celu stworzenie modelu uczenia maszynowego, który klasyfikuje wiadomości na prawdziwe i fałszywe. Zastosowane metody obejmują m.in. algorytmy **BERT**, **RoBERTa** oraz **Sentence Transformers**. Dane pochodzą z zestawu **ISOT Fake News Dataset**, zawierającego ponad 40 000 przykładów wiadomości. Projekt pozwala analizować różne podejścia do przetwarzania tekstu i oceny efektywności modeli.

---

## Wymagania

Aby uruchomić projekt, wymagane jest:
- **Python 3.x**
- **pip** (menedżer pakietów dla Pythona)

W celu weryfikacji środowiska wpisz w terminalu:

```bash
python --version
pip --version
```

---

## Instalacja

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/Foxxich/magisterka.git
   cd fake-news-detection
   ```

2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt
   ```

---

## Uruchamianie

1. **Przygotowanie danych**  
   Upewnij się, że pliki `Fake.csv` i `True.csv` znajdują się w folderze `datasets`.

2. **Uruchomienie skryptu**  
   W konsoli uruchom plik `main.py`:
   ```bash
   python main.py
   ```

3. **Wybór konfiguracji**  
   Postępuj zgodnie z instrukcjami w konsoli, aby wybrać reprezentację kontekstową, metodę podziału danych i algorytm.

---

## Pliki Projektu

- **`main.py`**  
  Główny plik uruchamiający cały projekt. Pozwala wybrać metodę reprezentacji kontekstowej, podziału danych oraz uruchamiać różne algorytmy opisane w dedykowanych plikach i folderach. Obsługuje 20 różnych metod uczenia oraz różne konfiguracje danych.

- **`common.py`**  
  Zawiera funkcje pomocnicze, takie jak:
  - Ładowanie i wstępne przetwarzanie danych (m.in. `load_and_preprocess_data`).
  - Funkcje do generowania reprezentacji kontekstowych (BERT, RoBERTa, Sentence Transformers).
  - Funkcje do podziału danych (klasyczny, one-shot, few-shot).
  - Funkcja do oceny modeli na podstawie wybranych metryk.

- **`my_run.py`**  
  Zawiera niestandardowe autorskie implementacje metod uczenia, w tym meta-modele łączące predykcje z różnych klasyfikatorów (np. GradientBoosting, RandomForest, CatBoost). Zawiera również implementację ważonych sieci neuronowych.

- **`my_plots.py`**  
  Skrypt generujący wykresy dla wyników uzyskanych w poszczególnych eksperymentach. Obsługuje m.in. Precision-Recall curves oraz wykresy porównawcze dla metryk takich jak Accuracy, Precision, Recall itp.

- **`all_plots.py`**  
  Rozszerzony skrypt do generowania wykresów porównawczych i Precision-Recall z podziałem na różne konfiguracje eksperymentów (classic, one-shot, few-shot).

---

## Foldery

- **`article1` - `article16`**  
  Zawierają pliki specyficzne dla poszczególnych metod uczenia. Każdy folder jest odpowiedzialny za implementację jednej metody.

- **`datasets`**  
  Folder zawierający dane wejściowe do projektu. Wymagane pliki: `Fake.csv` i `True.csv` z ISOT Fake News Dataset.

- **`results`**  
  Wyniki eksperymentów, w tym pliki `.csv` z wynikami dla każdej metody i konfiguracji.

- **`plots`**  
  Zawiera wykresy wygenerowane przez skrypty `my_plots.py` i `all_plots.py`.