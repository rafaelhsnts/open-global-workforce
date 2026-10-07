# **Open Global Workforce & Tech Salaries 2026 Dataset**

O dataset analisado é o **Open Global Workforce & Tech Salaries 2026**, obtido da base de dados do Kaggle. O conjunto de dados conta com 12003 registros e 10 colunas, e as variáveis **salary_min_usd** e **salary_max_usd** são a base para obtenção da variável alvo.
  
link: **https://www.kaggle.com/datasets/saitejabandaruin/open-global-workforce-intelligence-2026**  

## **Dataset**

### Features:  
  
* job_id: Código de identificação único no formato alfanumérico, destinado a identificação da vaga.  
* job_title: Título do cargo ofertado.   
* company_name: Nome da empresa responsável pela oferta da vaga.  
* location: Local de atuação da vaga.  
* tech_stack: Tecnologias associadas aos requisitos da vaga.  
* source: Plataforma onde a vaga foi divulgada.  
* description: Descrição da vaga.  
* date_posted: Data de publicação da vaga.  

### **Variáveis utilizadas para criação do Target:**

* salary_min_usd: Menor valor da faixa salarial anunciada para a vaga, em USD.
* salary_max_usd: Maior valor da faixa salarial anunciada para a vaga, em USD.

### **Target:**

* salary_mean_usd: Valor médio da faixa salarial anunciada para a vaga, em USD. Calculada a partir das variáveis salary_min_usd e salary_max_usd da base original, criada durante o tratamento dos dados.

## **Objetivo**

O projeto tem por objetivo, treinar um modelo de rede neural de regressão, que estime o valor médio salarial ofertado a profissionais da área de tecnologia a partir de características presentes na base de dados.

## **Avaliação**

Foram testadas diferentes arquiteturas de redes neurais, com o objetivo de verificar qual delas apresenta o melhor desempenho. A arquitetura 128-64 apresentou os menores valores de MAE e RMSE e o maior valor de R² entre as arquiteturas avaliadas.

| Métrica  | Resultado    |
|----------|--------------|
| **R²**   | 0.9244       |
| **MAE**  | US$ 7922.84  |
| **RMSE** | US$ 11455.21 |

## **Tecnologias**

- joblib
- kagglehub
- matplotlib
- numpy
- pandas
- scikit-learn
- seaborn
- streamlit
- torch

## **Estrutura do projeto**
```text
.  
├── README.md  
├── apps/  
│   └── open_global_workforce_streamlit.py  
├── datas/  
├── models/  
├── notebooks/  
│   └── open_global_workforce_ml.ipynb  
└── requirements.txt  
```

## **Método de execução**

1. Clone o repositório:

    git clone

2. Instale as dependências:

    pip install -r requirements.txt

3. Abra o notebook no Jupyter ou no Colab

    notebooks/open_global_workforce_ml.ipynb

4. Execute a aplicação streamlit

    streamlit run ./apps/open_global_workforce_streamlit.py

## **Pipeline**

* Análise exploratória
* Padronização e geração de variáveis dummies
* Separação dos dados em treino e teste
* Normalização da variável alvo
* Criação dos tensores
* Configuração do DataLoader e do modelo de regressão
* Treinamento do modelo
* Avaliação do modelo