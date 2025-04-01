import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    classification_report
)
from train_model import load_data


def load_model(path: str) -> LogisticRegression:
    """
    Carrega um modelo treinado a partir do caminho especificado utilizando joblib.

    :param path: Caminho para o arquivo do modelo (ex: '../src/models/logistic_model.joblib').
    :return: Modelo treinado.
    """
    return load(path)


def evaluate_model(model: LogisticRegression, X: pd.DataFrame, y: pd.Series, show_confusion: bool = False) -> None:
    """
    Avalia o modelo de regressão logística com base em métricas de classificação e imprime os resultados.
    Opcionalmente, plota a matriz de confusão.

    :param model: Modelo de regressão logística treinado.
    :param X: DataFrame com os dados de entrada.
    :param y: Series com os rótulos verdadeiros.
    :param show_confusion: Se True, exibe a matriz de confusão.
    """
    # Predições e probabilidades
    y_pred = model.predict(X)
    y_probs = model.predict_proba(X)

    # Cálculo das métricas
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_probs[:, 1])
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y, y_probs[:, 1])
    ks = max(tpr - fpr)
    report = classification_report(y, y_pred)

    # Impressão das métricas
    print("=" * 75)
    print("Métricas do modelo".center(75))
    print("=" * 75)
    print(f"Acurácia: {accuracy:.2%}")
    print(f"AUC: {auc:.2%}")
    print(f"Coeficiente de Gini: {gini:.2%}")
    print(f"KS: {ks:.4f}")
    print(report)

    # Exibe a matriz de confusão se solicitado
    if show_confusion:
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.get_cmap('Blues'))
        plt.title("Matriz de Confusão")
        plt.show()


def main() -> None:
    # Carrega os dados de teste e validação
    data_paths = {
        'X_test': '../data/processed/X_test.ftr',
        'y_test': '../data/processed/y_test.ftr',
        'X_valid': '../data/processed/X_valid.ftr',
        'y_valid': '../data/processed/y_valid.ftr'
    }
    dfs = load_data(**data_paths)

    X_test = dfs['X_test']
    y_test = dfs['y_test']
    X_valid = dfs['X_valid']
    y_valid = dfs['y_valid']

    # Carrega o modelo treinado
    model = load_model('../src/models/logistic_model.joblib')

    # Avalia o modelo na base de teste
    print("Avaliação na base de teste:")
    evaluate_model(model, X_test, y_test, show_confusion=True)
    print("=" * 50)

    # Avalia o modelo na base de validação
    print("Avaliação na base de validação:")
    evaluate_model(model, X_valid, y_valid, show_confusion=True)


if __name__ == '__main__':
    main()
