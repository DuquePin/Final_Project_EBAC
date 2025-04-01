import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


def categorize_variable(variable: pd.Series, bins: int = 10) -> pd.Series:
    """
    Função para categorizar variável contínua.

    :param variable: Series do Pandas, a variável a ser categorizada
    :param bins: número de categorias a criar com base em intervalos de igual tamanho
    :return: Series do Pandas com a variável categorizada
    """
    labels = pd.qcut(variable, bins, duplicates='drop')
    cat_variable = pd.Series(labels, name=f'cat_{variable.name}')
    return cat_variable


def calculate_information_value(feature: pd.Series, target: pd.Series) -> float:
    """
    Calcula o Information Value (IV) para uma determinada variável explicativa com base na
    variável-alvo binária usando o Peso da Evidência (WoE).

    :param feature: Pandas Series representando a variável explicativa (feature)
    :param target: Pandas Series representando a variável alvo (target)
    :return: Information Value (IV)
    """
    # Categoriza a variável se ela for contínua
    if feature.dtype.kind in 'if':
        feature = categorize_variable(feature)

    # Cria um dataframe com a feature e target
    df = pd.DataFrame({'feature': feature, 'target': target})

    # Agrupa o dataframe com as classes de feature e realiza sua contagem
    grouped = df.groupby('feature', observed=False)['target'].agg(['count', 'sum'])
    grouped.columns = ['total', 'good']
    grouped['bad'] = grouped['total'] - grouped['good']

    # Evita divisão por zero
    grouped = grouped[(grouped['good'] > 0) & (grouped['bad'] > 0)]

    # Calcula distribuições
    good_dist = grouped['good'] / grouped['good'].sum()
    bad_dist = grouped['bad'] / grouped['bad'].sum()

    # Calcula Weight of Evidence (WoE)
    grouped['WoE'] = np.log(good_dist / bad_dist)

    # Calcula Information Value (IV)
    grouped['IV'] = (good_dist - bad_dist) * grouped['WoE']
    iv = grouped['IV'].sum()

    return iv


def classify_iv(iv: float) -> str:
    """
    Função para classificar o IV de acordo com a classificação de Naeem Siddiqi.
    :param iv: o valor de Information Value da variável.
    :return: string com a classificação de IV da variável
    """
    if iv < 0.02:
        return 'Useless'
    elif 0.02 <= iv < 0.1:
        return 'Weak'
    elif 0.1 <= iv < 0.3:
        return 'Medium'
    elif 0.3 <= iv < 0.5:
        return 'Strong'
    return 'Overfit'


def classify_vif(vif: float) -> str:
    """
    Função para classificar o VIF de acordo com a classificação mais usada de valores VIF.
    :param vif: o valor de Variance Inflation Factor da variável
    :return:  string com a classificação de VIF da variável
    """
    if vif < 5:
        return 'Low'
    elif 5 <= vif < 10:
        return 'Moderate'
    elif vif >= 10:
        return 'High'
    else:
        return 'Unidentified'


def calculate_iv_vif(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """
    Função para calcular e classificar IV e VIF das variáveis.

    :param features: Pandas DataFrame contendo as variáveis independentes (features)
    :param target: Pandas Series contendo a variável dependente (target)
    :return:
    """
    results = []

    # Calcula Information Value para cada feature
    for column in features.columns:
        iv = calculate_information_value(features[column], target)
        iv_class = classify_iv(iv)
        results.append({'Feature': column, 'IV': iv, 'IV_Classification': iv_class})

    # Converte variáveis categóricas para valores numéricos
    features_numeric = pd.get_dummies(features, drop_first=True).astype(np.int64)

    # Calcula Variance Inflation Factor
    vif_data = features_numeric.copy()
    vif_data['Intercept'] = 1
    vif_values = [
        (variance_inflation_factor(vif_data.values, i), classify_vif(variance_inflation_factor(vif_data.values, i)))
        for i in range(vif_data.shape[1])]

    vif_df = pd.DataFrame(vif_values, columns=['VIF', 'VIF_Classification'], index=vif_data.columns)

    # Remove intercepto dos results
    vif_df = vif_df.drop(index='Intercept')

    # Junta os results de IV e VIF em um DataFrame
    df_result = pd.DataFrame(results)
    df_result = df_result.merge(vif_df, left_on='Feature', right_index=True, how='left')

    return df_result


def transform_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Função para transformar os dados, padronizando variáveis númericas e codificando variáveis categóricas

    :param dataframe: recebe o Pandas DataFrame com os dados a transformar
    :return: retorna o Pandas DataFrame com os dados transformados
    """
    # Listando variáveis númericas
    numeric_features = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    # Criando objeto para padronizar variáveis númericas
    scaler = StandardScaler()

    # Padronizando variáveis númericas
    dataframe[numeric_features] = scaler.fit_transform(dataframe[numeric_features])

    # Aplicando One-hot encoding para variáveis categóricas
    dataframe = pd.get_dummies(dataframe, drop_first=True).astype(np.int64)

    return dataframe


def select_best_features(results: pd.DataFrame) -> list:
    """
    Função para selecionar os melhores features com base no cálculo de Information Value e Variance Inflation Factor

    :param results: recebe Pandas DataFrame com a tabela de resultados do cálculo de IV e VIF e suas classificações
    :return:
    """
    selected_features = []
    for index, row in results.iterrows():
        if row['IV_Classification'] in ['Overfit', 'Strong', 'Medium', 'Weak'] and row['VIF_Classification'] in ['Low',
                                                                                                                 'Moderate']:
            selected_features.append(row['Feature'])
    return selected_features


def split_data(dataframe: pd.DataFrame) -> tuple:
    """
    Função para dividir os dados em validação, treino e teste.

    :param dataframe: recebe Pandas DataFrame com os dados
    :return: tupla com os dados separados
    """
    # Criando base de validação (5%) e desenvolvimento (95%)
    df_valid = dataframe.sample(frac=.05, random_state=412)
    df_dev = dataframe.drop(df_valid.index)

    # Separando em Features (X) e Target (y)
    X, y = df_dev.drop(columns=['Target']), df_dev.Target
    X_valid, y_valid = df_valid.drop(columns=['Target']), df_valid.Target

    # Separando em dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.25, random_state=412)

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def save_data(path: str, X_train: pd.DataFrame, X_test: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series,
              y_test: pd.Series, y_valid: pd.Series) -> None:
    """
    Função para salvar os dados de treino, teste e validação no caminho específicado.

    :param path: string contendo o caminho onde os dados serão salvos
    :param X_train: Pandas DataFrame com as variáveis explicativas de treino (features)
    :param X_test: Pandas DataFrame com as variáveis explicativas de teste (features)
    :param X_valid: Pandas DataFrame com as variáveis explicativas de validação (features)
    :param y_train: Pandas Series com a variável-alvo de treino (target)
    :param y_test: Pandas Series com a variável-alvo de test (target)
    :param y_valid: Pandas Series com a variável-alvo de validação (target)
    :return: None
    """
    X_train.to_feather(path+'X_train.ftr')
    X_test.to_feather(path+'X_test.ftr')
    X_valid.to_feather(path+'X_valid.ftr')
    y_train.to_frame().to_feather(path+'y_train.ftr')
    y_test.to_frame().to_feather(path+'y_test.ftr')
    y_valid.to_frame().to_feather(path+'y_valid.ftr')


def main():
    # Importando dados
    df = pd.read_csv('../data/raw/train.csv')

    # Mapeando variável Target
    df['Target'] = df['Target'].map({'Graduate': 0, 'Enrolled': 0, 'Dropout': 1})

    # Transformando dados
    df = transform_data(df)

    # Selecionando melhores features
    results = calculate_iv_vif(df.drop(columns=['Target']), df.Target)
    selected_features = select_best_features(results)

    # Criando subconjunto dos dados com os melhores features
    df_sub = df[selected_features].copy()
    df_sub['Target'] = df['Target'].copy()

    # Separando dados em Validação, Treino e Teste
    X_train, X_test, X_valid, y_train, y_test, y_valid = split_data(df_sub)

    # Salvando dados
    save_data('../data/processed/', X_train, X_test, X_valid, y_train, y_test, y_valid)

    # Visualizando resultado
    print('Arquivo preprocessing.py rodado com sucesso!')


if __name__ == '__main__':
    main()
