import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def load_data(**paths):
    """
    Carrega arquivos a partir dos caminhos especificados, utilizando o formato Feather.

    :param paths: Parâmetros nomeados com os caminhos dos arquivos a serem carregados.
    :return: Dicionário com os DataFrames carregados.
    """
    dfs = {}
    for name, path in paths.items():
        dfs[name] = pd.read_feather(path)
    return dfs


def apply_grid_search(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """
    Aplica o GridSearchCV para encontrar os melhores hiperparâmetros do modelo de regressão logística.

    :param X_train: DataFrame com os dados de treinamento.
    :param y_train: Series com a variável alvo dos dados de treinamento.
    :return: Objeto GridSearchCV já ajustado aos dados.
    """
    # Cria o modelo de regressão logística
    log_reg = LogisticRegression(max_iter=5000, random_state=412)

    # Define os parâmetros a serem testados
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    # Configura o GridSearchCV com 5 folds e múltiplas métricas de avaliação
    grid_search = GridSearchCV(log_reg, param_grid, cv=5,
                               scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                               refit=False,
                               verbose=3,
                               n_jobs=-1)

    # Ajusta o grid search aos dados de treino
    grid_search.fit(X_train, y_train)

    return grid_search


def train_model(params: dict, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Treina o modelo de regressão logística com os melhores parâmetros encontrados.

    :param params: Dicionário contendo os melhores parâmetros.
    :param X_train: DataFrame com os dados de treinamento.
    :param y_train: Series com a variável alvo dos dados de treinamento.
    :return: Modelo treinado.
    """
    # Cria um modelo de regressão logística com os parâmetros otimizados
    logistic = LogisticRegression(max_iter=1000, random_state=412, **params)

    # Ajusta o modelo aos dados de treinamento
    logistic.fit(X_train, y_train)

    return logistic


def save_model(path: str, model: LogisticRegression) -> None:
    """
    Salva o modelo treinado em disco utilizando joblib.

    :param path: Caminho (incluindo o nome do arquivo) onde o modelo será salvo.
    :param model: Modelo treinado a ser salvo.
    :return: None
    """
    dump(model, path + '.joblib')


def main():
    # Carrega os dados de treinamento
    dfs = load_data(
        X_train='../data/processed/X_train.ftr',
        y_train='../data/processed/y_train.ftr',
    )

    X_train = dfs['X_train']
    y_train = dfs['y_train']

    # Aplica GridSearchCV para encontrar os melhores hiperparâmetros
    grid = apply_grid_search(X_train, y_train['Target'])

    # Converte os resultados do grid search para um DataFrame
    results = pd.DataFrame(grid.cv_results_)

    # Calcula a média dos ranks das métricas de avaliação
    results['average_rank'] = results[
        ['rank_test_accuracy', 'rank_test_precision', 'rank_test_recall', 'rank_test_f1', 'rank_test_roc_auc']].mean(
        axis=1)

    # Seleciona os melhores parâmetros com base na maior média dos ranks
    best_params = results.loc[results['average_rank'].idxmax(), 'params'].copy()

    # Treina o modelo utilizando os melhores parâmetros encontrados
    model = train_model(best_params, X_train, y_train['Target'])

    # Salva o modelo treinado
    save_model('../src/models/logistic_model', model)


if __name__ == '__main__':
    main()
