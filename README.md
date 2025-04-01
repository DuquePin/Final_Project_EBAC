# Projeto Final EBAC
Este projeto faz parte dos requisitos para a ingressão no programa de empregabilidade do curso
**Profissão: Cientista de Dados** da **Escola Britânica de 
Artes Criativas & Tecnologia (EBAC)**.

## Objetivo
O objetivo do projeto é identificar uma problemática do mundo real que possa ser solucionada por meio da **análise de 
dados** e do uso de **Machine Learning**. A proposta é demonstrar a relevância dos dados na construção de uma solução 
eficaz  e embasada.

## Problemática
A problemática escolhida para o projeto é a evasão de alunos no ensino superior. No Brasil, esse fenômeno tem se tornado 
uma tendência preocupante. De acordo com o **14º Mapa do Ensino Superior no Brasil – 2024**, publicado pelo Instituto 
Semesp, a taxa média de evasão no ensino superior brasileiro é de 57,2%. [[fonte]](https://www.semesp.org.br/mapa/edicao-14/brasil/evasao/?utm_source=chatgpt.com)

Diante desse cenário, o objetivo do projeto é desenvolver um modelo preditivo baseado em Machine Learning que seja capaz 
de analisar dados históricos e identificar padrões para prever quais alunos apresentam maior risco de abandono. Com essa 
previsão, será possível implementar ações preventivas, permitindo que instituições de ensino adotem medidas como suporte 
acadêmico, financeiro ou psicológico para reduzir as taxas de evasão e melhorar a retenção estudantil.

## Base de Dados

Os dados deste projeto foram extraídos da competição **Kaggle** intitulada *Classification with an Academic Success 
Dataset.* Essa competição utiliza um conjunto de dados gerado por um modelo de aprendizagem profunda treinado no dataset 
original *Predict Students' Dropout and Academic Success.*

Para mais informações, acesse os links abaixo:
- [Página da competição no Kaggle](https://www.kaggle.com/competitions/playground-series-s4e6/data)
- [Conjunto de dados original](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)



## Estrutura
O projeto foi organizado em quatro notebooks do Jupyter, cada um abordando uma etapa específica da análise e modelagem dos dados:

- EDA (Exploratory Data Analysis): Apresenta a contextualização do problema e a análise exploratória dos dados.
- Preprocessing: Responsável pelo pré-processamento dos dados com base nos insights obtidos na análise exploratória.
- Modeling: Foca na construção do modelo preditivo, utilizando Regressão Logística em duas abordagens distintas.
- Evaluation: Avalia o desempenho dos modelos criados, comparando os resultados para determinar a abordagem mais eficaz.

Além dos notebooks, foi desenvolvida uma aplicação web em Streamlit, que permite visualizar os resultados da modelagem 
utilizando os modelos previamente treinados. Abaixo, segue um vídeo demonstrativo da aplicação. 

Por fim, todos os scripts foram adaptados para uso local e estão disponíveis na pasta do projeto, juntamente com os 
arquivos de requisitos necessários para a execução.