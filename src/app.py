import streamlit as st
import re
import pickle
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)


@st.cache_data
def load_data(file_path, file_type):
    """
    Função para carregar dados a partir do tipo do arquivo passado.

    :param file_path: caminho do arquivo
    :param file_type: tipo do arquivo
    :return:
    """
    try:
        if file_type == 'ftr':
            return pd.read_feather(file_path)
        elif file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'xlsx':
            return pd.read_excel(file_path)
        else:
            raise ValueError('Tipo de arquivo não suportado.')
    except Exception as exc:
        st.error(f'Erro ao carregar os dados: {exc}')
        return None


@st.cache_data
def load_model(model_path, file_type):
    """
    Função para carregar modelo pré-treinado a partir do tipo do arquivo passado.

    :param model_path: caminho do arquivo
    :param file_type: tipo do arquivo
    :return:
    """
    try:
        if file_type == 'pkl':
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            return model
        elif file_type == 'joblib':
            model = joblib.load(model_path)
            return model
        else:
            raise ValueError('Tipo de arquivo não suportado.')
    except Exception as exc:
        st.error(f'Erro ao carregar o modelo: {exc}')
        return None


def evaluate_model(_model, X_test, y_test):
    """
    Função para avaliar o modelo com os dados de teste.

    :param _model: Modelo treinado
    :param X_test: Dados de teste, as features da base
    :param y_test: Dados de teste, a target da base
    :return:
    """
    try:
        # Fazendo previsões
        y_pred = _model.predict(X_test)
        y_proba = _model.predict_proba(X_test)[:, 1]

        # Calculando as métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        gini = 2 * auc - 1
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = max(tpr - fpr)

        # Métricas
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC Score': auc,
            'Gini': gini,
            'KS': ks
        }

        # Plotando a matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Não Evasão', 'Evasão'],
                    yticklabels=['Não Evasão', 'Evasão'])
        plt.xlabel('Predição')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        st.pyplot(fig)

        return metrics
    except Exception as e:
        st.error(f'Erro na avaliação do modelo: {e}')
        return {}


def main():
    # Configurações da página
    st.set_page_config(page_title='Evasão Escolar', layout='wide')
    st.title('Classificação de Evasão do Ensino Superior')

    # Explicação básica
    st.subheader('Aplicação')
    st.write('''O objetivo desta aplicação é avaliar modelos pré-treinados com base em dados previamente transformados. 
    Para isso, assume-se que os dados foram devidamente preparados e que o modelo foi treinado de forma adequada. 
    Os códigos para preparar os dados e treinar o modelo estão disponíveis nas pastas do projeto.
    ''')

    # Upload de dados
    fp_data = st.file_uploader('Carregue a base de dados transformada:', type=['ftr', 'csv', 'xlsx'])
    fp_model = st.file_uploader('Carregue o modelo treinado:', type=['pkl', 'joblib'])

    if fp_data and fp_model:
        match_data = re.search(r'\.([a-zA-Z0-9.]+)$', fp_data.name)
        match_model = re.search(r'\.([a-zA-Z0-9.]+)$', fp_model.name)

        if match_data and match_model:
            file_ext_data = match_data.group(1)
            file_ext_model = match_model.group(1)

            # Carregar base de dados
            df = load_data(fp_data, file_ext_data)

            # Carregar modelo treinado
            model = load_model(fp_model, file_ext_model)

            if df is not None and model is not None:
                st.success('Base de dados e modelo carregados com sucesso!')
                st.write(df.head(5))

                if 'Target' not in df.columns:
                    st.error('A coluna alvo deve ser nomeada \'Target\'.')
                else:
                    # Separando as features e a target
                    X = df.drop(columns=['Target'])
                    y = df['Target']

                    # Avaliando o modelo
                    st.write('### Avaliação do Modelo:')
                    metrics = evaluate_model(model, X, y)
                    accuracy, precision, recall, f1, auc, gini, ks = metrics.values()
                    st.write(f'Acurácia do Modelo: {accuracy:.2%}')
                    st.write(f'Precisão da Classe 1 do Modelo: {precision:.2%}')
                    st.write(f'Recall da Classe 1 do Modelo: {recall:.2%}')
                    st.write(f'F1 da Classe 1 do Modelo: {f1:.2%}')
                    st.write(f'AUC do Modelo: {auc:.2%}')
                    st.write(f'Gini do Modelo: {gini:.2%}')
                    st.write(f'KS do Modelo: {ks:.2f}')
            else:
                st.error('Erro ao carregar os dados ou o modelo.')


if __name__ == '__main__':
    main()
