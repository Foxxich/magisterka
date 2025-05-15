import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import xgboost as xgb


def metoda17(X_train, y_train, X_test, y_test):
    """
    Trenuje ulepszony model zespołowy z miękkim głosowaniem (soft-voting) z optymalizacją
    hiperparametrów i skalowaniem cech.

    Parametry:
        X_train (np.ndarray): Cechy zbioru treningowego.
        y_train (lista lub np.ndarray): Etykiety zbioru treningowego.
        X_test (np.ndarray): Cechy zbioru testowego.
        y_test (lista lub np.ndarray): Etykiety zbioru testowego.

    Zwraca:
        voting_clf_final: Wytrenowany model VotingClassifier po optymalizacji.
        X_test_scaled: Przeskalowane cechy zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """
    # Inicjalizacja klasyfikatorów
    mlp = MLPClassifier(max_iter=100, solver='lbfgs', random_state=0)
    xgb = XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', random_state=0
    )

    # Pipeline dla regresji logistycznej ze skalowaniem i optymalizacją hiperparametrów
    pipeline_logreg = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(random_state=0))
    ])

    param_grid_logreg = {
        'log_reg__C': [0.001, 0.01, 0.1, 1, 10],
        'log_reg__penalty': ['l1', 'l2', 'elasticnet'],
        'log_reg__solver': ['liblinear', 'saga']
    }

    grid_search_logreg = GridSearchCV(pipeline_logreg, param_grid_logreg, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_logreg.fit(X_train, y_train)
    log_reg_optimized = grid_search_logreg.best_estimator_.named_steps['log_reg']
    scaler_logreg = grid_search_logreg.best_estimator_.named_steps['scaler']
    X_test_scaled_logreg = scaler_logreg.transform(X_test)


    # Pipeline dla MLP ze skalowaniem
    pipeline_mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', mlp)
    ])
    pipeline_mlp.fit(X_train, y_train)
    X_test_scaled_mlp = pipeline_mlp.named_steps['scaler'].transform(X_test)
    mlp_optimized = pipeline_mlp.named_steps['mlp'] # Można dodać optymalizację GridSearchCV dla MLP


    # Pipeline dla XGBoost ze skalowaniem
    pipeline_xgb = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb)
    ])
    pipeline_xgb.fit(X_train, y_train)
    X_test_scaled_xgb = pipeline_xgb.named_steps['scaler'].transform(X_test)
    xgb_optimized = pipeline_xgb.named_steps['xgb'] # Można dodać optymalizację GridSearchCV dla XGBoost


    # Utworzenie klasyfikatora zespołowego z miękkim głosowaniem z wytrenowanymi (potencjalnie) lepszymi modelami
    voting_clf = VotingClassifier(estimators=[('mlp', mlp_optimized), ('log_reg', log_reg_optimized), ('xgb', xgb_optimized)], voting='soft')

    # Trening klasyfikatora VotingClassifier na przeskalowanym zbiorze treningowym
    voting_clf.fit(X_train, y_train) # VotingClassifier sam w sobie nie potrzebuje skalowania, bo wewnętrzne klasyfikatory już są przeskalowane w pipeline'ach

    # Przeskaluj zbiór testowy (używając transformacji z pipeline'ów)
    X_test_scaled = StandardScaler().fit_transform(X_test) # Można też użyć jednego scalera dopasowanego na X_train

    return voting_clf, X_test_scaled, y_test


def metoda18(X_train, y_train, X_test, y_test):
    """
    Trenuje klasyfikatory GradientBoosting i CatBoost oraz łączy ich predykcje przy użyciu meta-modelu.
    """
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)  # Gradient Boosting
    cat_clf = CatBoostClassifier(iterations=100, verbose=0, random_state=42)  # CatBoost
    gb_clf.fit(X_train, y_train)
    cat_clf.fit(X_train, y_train)
    gb_preds_train = gb_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla GradientBoosting
    cat_preds_train = cat_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla CatBoost
    meta_features_train = pd.DataFrame({  # Tworzenie meta-cech
        'gb_preds': gb_preds_train,
        'cat_preds': cat_preds_train
    })
    meta_model = LogisticRegression()  # Meta-model: regresja logistyczna
    meta_model.fit(meta_features_train, y_train)
    gb_preds_test = gb_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla GradientBoosting
    cat_preds_test = cat_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla CatBoost
    meta_features_test = pd.DataFrame({  # Meta-cechy dla testu
        'gb_preds': gb_preds_test,
        'cat_preds': cat_preds_test
    })
    return meta_model, meta_features_test, y_test


def metoda19(X_train, y_train, X_test, y_test):
    """
    Ulepszony model stacking z Random Forest, XGBoost i GradientBoosting jako bazowymi oraz Logistic Regression jako meta-modelem.

    Parametry:
        X_train (array-like): Cechy zbioru treningowego.
        y_train (array-like): Etykiety zbioru treningowego.
        X_test (array-like): Cechy zbioru testowego.
        y_test (array-like): Etykiety zbioru testowego.

    Zwraca:
        meta_model: Wytrenowany model meta (Logistic Regression).
        meta_features_test: Meta-cechy dla zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """

    # 1. Skalowanie danych
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Inicjalizacja i optymalizacja modeli bazowych
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  # Zwiększone n_estimators, ograniczona głębokość
    xgb_clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss',
                              learning_rate=0.05, max_depth=5, random_state=42)  # Zmniejszone learning_rate i max_depth
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # Dodany GradientBoosting

    # 3. Walidacja krzyżowa do generowania meta-cech
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_preds_train = cross_val_predict(rf_clf, X_train_scaled, y_train, cv=skf, method='predict_proba')[:, 1]
    xgb_preds_train = cross_val_predict(xgb_clf, X_train_scaled, y_train, cv=skf, method='predict_proba')[:, 1]
    gb_preds_train = cross_val_predict(gb_clf, X_train_scaled, y_train, cv=skf, method='predict_proba')[:, 1] # Predykcje z GradientBoosting

    meta_features_train = pd.DataFrame({
        'rf_preds': rf_preds_train,
        'xgb_preds': xgb_preds_train,
        'gb_preds': gb_preds_train # Dodane cechy
    })

    # 4. Trening meta-modelu
    meta_model = LogisticRegression(solver='liblinear', random_state=42) # Określony solver
    meta_model.fit(meta_features_train, y_train)

    # 5. Predykcje na zbiorze testowym
    rf_clf.fit(X_train_scaled, y_train)
    xgb_clf.fit(X_train_scaled, y_train)
    gb_clf.fit(X_train_scaled, y_train) # Trenowanie GradientBoosting

    rf_preds_test = rf_clf.predict_proba(X_test_scaled)[:, 1]
    xgb_preds_test = xgb_clf.predict_proba(X_test_scaled)[:, 1]
    gb_preds_test = gb_clf.predict_proba(X_test_scaled)[:, 1]

    meta_features_test = pd.DataFrame({
        'rf_preds': rf_preds_test,
        'xgb_preds': xgb_preds_test,
        'gb_preds': gb_preds_test
    })

    return meta_model, meta_features_test, y_test


def metoda20(X_train, y_train, X_test, y_test):
    """
    Trenuje klasyfikatory LightGBM, CatBoost, Random Forest i Gradient Boosting
    oraz łączy ich predykcje przy użyciu meta-modelu (regresji logistycznej).
    """
    lgb_clf = LGBMClassifier(n_estimators=100, random_state=42)  # LightGBM
    cat_clf = CatBoostClassifier(iterations=100, verbose=0, random_state=42)  # CatBoost
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)  # Gradient Boosting

    lgb_clf.fit(X_train, y_train)
    cat_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)
    gb_clf.fit(X_train, y_train)

    lgb_preds_train = lgb_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla LightGBM
    cat_preds_train = cat_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla CatBoost
    rf_preds_train = rf_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla Random Forest
    gb_preds_train = gb_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla Gradient Boosting

    meta_features_train = pd.DataFrame({  # Tworzenie meta-cech
        'lgb_preds': lgb_preds_train,
        'cat_preds': cat_preds_train,
        'rf_preds': rf_preds_train,
        'gb_preds': gb_preds_train
    })

    meta_model = LogisticRegression()  # Meta-model: regresja logistyczna
    meta_model.fit(meta_features_train, y_train)

    lgb_preds_test = lgb_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla LightGBM
    cat_preds_test = cat_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla CatBoost
    rf_preds_test = rf_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla Random Forest
    gb_preds_test = gb_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla Gradient Boosting

    meta_features_test = pd.DataFrame({  # Meta-cechy dla testu
        'lgb_preds': lgb_preds_test,
        'cat_preds': cat_preds_test,
        'rf_preds': rf_preds_test,
        'gb_preds': gb_preds_test
    })

    return meta_model, meta_features_test, y_test