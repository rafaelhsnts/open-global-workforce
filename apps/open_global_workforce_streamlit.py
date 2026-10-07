import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import sys
import os
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.header("Open Global Workforce", text_alignment="center")

architectures = joblib.load("./models/architectures.pkl")

df = pd.read_csv("./datas/global_tech_market_2026.csv")
df_treated = pd.read_csv("./datas/global_tech_market_2026_treated.csv")

X = df_treated.loc[:, df_treated.columns != "salary_mean_usd"]
y = df_treated[["salary_mean_usd"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)

class OpenGlobalWorkforceModel(nn.Module):
    def __init__(self, input_size, output_size, architecture):
        super(OpenGlobalWorkforceModel, self).__init__()
        layers = []
        previous_size = input_size
        for neurons in architecture:
            layers.append(nn.Linear(previous_size, neurons))
            layers.append(nn.ReLU())
            previous_size = neurons
        layers.append(nn.Linear(previous_size, output_size))
        self.model = nn.Sequential(*layers)
    def forward(self, X):
        return self.model(X)

menu = st.sidebar.radio("Menu", ["Modelos", "Descrição", "Dados"])

if menu == "Modelos":
    st.subheader("Modelos", text_alignment="center")
    with st.container(border=True):
        selected_architecture = st.segmented_control("**Seleção de modelo:**", architectures, required=True, default="32", width="stretch")

        checkpoint = torch.load(f"./models/{selected_architecture}/model.pth", weights_only=False)
        model = OpenGlobalWorkforceModel(input_size=checkpoint["input_size"], output_size=checkpoint["output_size"], architecture=architectures[selected_architecture])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        job = df["job_title"].str.lower().str.split(",").explode().str.strip().str.replace(" ", "_").unique().tolist()
        df.loc[df["location"].str.lower().str.startswith("remote"), "location"] = "remote"
        local = df["location"].str.lower().str.split(",").str[-1].str.strip().str.replace(" ", "_").unique().tolist()
        techs = df["tech_stack"].str.lower().str.split(",").explode().str.strip().str.replace(" ", "_").unique().tolist()

        job_display = [x.replace("_", " ").title() for x in job]
        job_selected = st.selectbox("Cargo", job_display)
        job_encoded = {x: int(x == job_selected) for x in job_display}

        local_display = [x.replace("_", " ").title() for x in local]
        local_selected = st.selectbox("Localidade", local_display)
        local_encoded = {x: int(x == local_selected) for x in local_display}

        techs_display = [x.replace("_", " ").title() for x in techs]
        techs_selected = st.multiselect("Tecnologias", techs_display)
        techs_selected = [techs[techs_display.index(x)] for x in techs_selected]
        techs_encoded = {x: int(x in techs_selected) for x in techs}

        entrada = pd.DataFrame([{**job_encoded, **local_encoded, **techs_encoded}])
        features = checkpoint["features"]
        entrada = entrada.reindex(columns=features, fill_value=0)

        confirmar = st.button("Confirmar", use_container_width=True)
        if confirmar:
            if not techs_selected:
                st.info("Selecione pelo menos uma tecnologia.")
            else:
                entrada_np = entrada.to_numpy(dtype=np.float32)
                entrada_torch = torch.from_numpy(entrada_np)
                with torch.no_grad():
                    pred = model(entrada_torch)
                pred = checkpoint["y_scaler"].inverse_transform(pred.numpy())
                salary = f"US$ {float(pred[0, 0]):.2f}"
                st.markdown(f"""<div style="
                background-color:#99e;
                color:#eee;
                text-align:center;
                font-weight:bold;
                font-size:30px;
                padding:5px;
                margin-bottom:15px;
                border-radius:5px
                ">Salário médio previsto:<br>{salary}</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:10px; color:#888; text-align:justify">
        <span>
            O modelo foi treinado com base em combinações de tecnologias consideradas reais e plausíveis. A seleção de tecnologias aleatórias ou em quantidades excessivas, que fogem ao que seria esperado para uma vaga real, pode resultar em previsões menos confiáveis.
        </span>
        <br>
        <br>
        <span>
            Esta plataforma tem finalidade exclusivamente educacional.
        </span>
    </div>
    """, unsafe_allow_html=True)

elif menu == "Descrição":
    with st.container(border=True):
        st.markdown("""
            ##### **Descrição:**

            O dataset analisado é o **Open Global Workforce & Tech Salaries 2026**, obtido da base de dados do kaggle. O conjunto de dados conta com 12003 registros e 10 colunas, e as variáveis **salary_min_usd** e **salary_max_usd** serão a base para obtenção da variável alvo.
            
            **link:** https://www.kaggle.com/datasets/saitejabandaruin/open-global-workforce-intelligence-2026
        """)

    with st.container(border=True):
        st.markdown("""
            ##### **Features:**
            
            * **job_id**: Código de identificação único no formato alfanumérico, destinado a identificação da vaga.
            * **job_title**: Título do cargo ofertado.
            * **company_name**: Nome da empresa responsável pela oferta da vaga.
            * **location**: Local de atuação da vaga.
            * **tech_stack**: Tecnologias associadas aos requisitos da vaga.
            * **source**: Plataforma onde a vaga foi divulgada.
            * **description**: Descrição da vaga.
            * **date_posted**: Data de publicação da vaga.  
        """)

    with st.container(border=True):
        st.markdown("""
            ##### **Variáveis utilizadas para criação do Target:**

            * **salary_min_usd**: Menor valor da faixa salarial anunciada para a vaga, em USD.
            * **salary_max_usd**: Maior valor da faixa salarial anunciada para a vaga, em USD.
        """)

    with st.container(border=True):
        st.markdown("""
            ##### **Target:**

            * **salary_mean_usd:** Valor médio da faixa salarial anunciada para a vaga, em USD. Calculada a partir das variáveis salary_min_usd e salary_max_usd da base original, criada durante o tratamento dos dados.
        """)

    with st.container(border=True):
        st.markdown("""
            ##### **Objetivo:**

            O projeto tem por objetivo, treinar um modelo de rede neural de regressão, que estime o valor médio salarial ofertado a profissionais da área de tecnologia a partir de características presentes na base de dados.  
        """)

    with st.container(border=True):
        st.markdown(f"""
            ##### **Conclusão:**

            Este projeto teve como objetivo inicial, a criação de um modelo de regressão linear que pudesse identificar a média salarial de profissionais da tecnologia, a partir de dados característicos da sua profissão. O modelo inicialmente possuía 10 variáveis advindas diretamente da base de dados obtida na plataforma do Kaggle, onde 2 dessas variáveis **salary_min_usd** e **salary_max_usd**, servem de base para obtenção da variável resposta **salary_mean_usd**, em contrapartida, as 8 variáveis restantes fornecem as 64 novas variáveis de entrada do modelo. 

            Com uma análise aprofundada nos dados, identificou-se entre os registros da variável **company_name**, nomes de empresas conhecidos dentro da cultura pop, indicando que possivelmente foram retirados de uma obra de ficção, como **Umbrella Corp**, **Stark Industries**, **Wayne Enterprises** e **CyberDyne**, o que sugere que a base passou por um processo de aumento artificial de dados, ou que toda ela tenha sido criada artificialmente. Por fim, para melhorar a otimização dos dados, bem como facilitar a obtenção dos pesos, e ao mesmo tempo obter resultados em uma escala mais agradável para visualização, os dados da variável de saída foram normalizados com o uso do **StandardScaler**.

            Com o treino e avaliação do modelo, e com as análises realizadas ao longo do projeto identificou-se que um modelo simples de regressão, não se ajustaria bem as complexidades nas relações entre as features e a label do modelo, sinalizando uma provável relação não linear entre elas. Na tentativa de obter melhores resultados, optou-se pela adição de camadas ocultas, buscando aumentar a capacidade do modelo de encontrar padrões não lineares presentes nos dados, transformando-o de uma regressão linear simples para uma rede neural de regressão.  

            Após a mudança, foram adicionados 5 modelos de arquiteturas de redes neurais a serem treinados, e comparando os resultados de cada uma delas, pôde-se identificar quais melhor respondiam às features de entrada. Dentre as arquiteturas treinadas, a que melhor respondeu a base de dados e retornou métricas mais favoráveis, foi a de **128-64** que dispunha de 2 camadas ocultas estimuladas por uma função de ativação ReLU. como resultado obteve-se um R² de 0.9244, um erro MAE de US$7922.84 e um erro RMSE de US$11455.21.
        """)

elif menu == "Dados":
    submenu = st.sidebar.radio("Análises", ["Dataframe", "Resultados"])
    if submenu == "Dataframe":
        dfot = st.segmented_control("**Seleção de DataFrame**", ["DataFrame Original", "DataFrame Tratado"], width="stretch", default="DataFrame Original")
        if dfot == "DataFrame Original":
            with st.container(border=True):
                st.subheader("DataFrame Original")
                st.dataframe(df, hide_index=True, use_container_width=True)

            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.markdown(f"""<div style="margin-bottom:15px; text-align:center; font-size:30px">{df.shape[0]} Registros</div>""", unsafe_allow_html=True)
                with col2:
                    with st.container(border=True):
                        st.markdown(f"""<div style="margin-bottom:15px; text-align:center; font-size:30px">{df.shape[1]} Colunas</div>""", unsafe_allow_html=True)

            with st.container(border=True):
                st.subheader("Tipos de dados")
                features_df = pd.DataFrame({
                    "Feature": df.columns,
                    "Tipo": df.dtypes.astype(str),
                    "Valores únicos": df.nunique().values
                })
                st.dataframe(features_df, hide_index=True, use_container_width=True)
                
        if dfot == "DataFrame Tratado":
            with st.container(border=True):
                st.subheader("DataFrame Tratado")
                st.dataframe(df_treated, hide_index=True, use_container_width=True)

            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    with st.container(border=True):
                        st.markdown(f"""<div style="margin-bottom:15px; text-align:center; font-size:25px">{X.shape[0]} Registros</div>""", unsafe_allow_html=True)
                with col2:
                    with st.container(border=True):
                        st.markdown(f"""<div style="margin-bottom:15px; text-align:center; font-size:25px">{X.shape[1]} Features</div>""", unsafe_allow_html=True)
                with col3:
                    with st.container(border=True):
                        st.markdown(f"""<div style="margin-bottom:15px; text-align:center; font-size:25px">{y.shape[1]} Target</div>""", unsafe_allow_html=True)

            with st.container(border=True):
                st.subheader("Tipos das Features")
                features_df_treated = pd.DataFrame({
                    "Feature": X.columns,
                    "Tipo": X.dtypes.astype(str),
                    "Valores únicos": X.nunique().values
                })
                st.dataframe(features_df_treated, hide_index=True, use_container_width=True)
                st.subheader("Tipo da Target")
                target_df_treated = pd.DataFrame({
                    "Feature": y.columns,
                    "Tipo": y.dtypes.astype(str),
                    "Valores únicos": y.nunique().values
                })
                st.dataframe(target_df_treated, hide_index=True, use_container_width=True)
    elif submenu == "Resultados":
        metrics = st.sidebar.radio("Métricas", ["Treino", "Avaliação"])
        selected_architecture = st.segmented_control("**Seleção de modelo:**", architectures, required=True, default="32", width="stretch")

        checkpoint = torch.load(f"./models/{selected_architecture}/model.pth", weights_only=False)
        model = OpenGlobalWorkforceModel(input_size=checkpoint["input_size"], output_size=checkpoint["output_size"], architecture=architectures[selected_architecture])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if metrics == "Treino":

            st.subheader("Métricas de treino")
            with st.container(border=True):
                fig = plt.figure(figsize=(10, 5))
                sns.lineplot(checkpoint["losses"])
                plt.title("Loss x Epoch")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                st.pyplot(fig)

            with st.container(border=True):
                fig = plt.figure(figsize=(10, 5))
                weights = np.array(checkpoint["weights"])
                for i in range(weights.shape[1]):
                    sns.lineplot(x=range(len(weights)), y=weights[:,i]) 
                plt.title("Weight x Epoch")
                plt.xlabel("epoch")
                plt.ylabel("weight")
                st.pyplot(fig, clear_figure=True)

            with st.container(border=True):
                fig = plt.figure(figsize=(10, 5))
                sns.lineplot(checkpoint["bias"]) 
                plt.title("Bias x Epoch")
                plt.xlabel("epoch")
                plt.ylabel("bias")
                st.pyplot(fig, clear_figure=True)

        if metrics == "Avaliação":
            st.subheader("Métricas de avaliação")
            X_test_torch = torch.tensor(X_test.to_numpy(dtype=np.float32))
            y_test_scaled = checkpoint["y_scaler"].transform(y_test)

            with torch.no_grad():
                y_pred_scaled = model(X_test_torch)
            y_pred_scaled = y_pred_scaled.numpy()
            y_pred = checkpoint["y_scaler"].inverse_transform(y_pred_scaled)

            y_test_original = checkpoint["y_scaler"].inverse_transform(y_test_scaled)

            mae = mean_absolute_error(y_test_original, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
            r2 = r2_score(y_test_original, y_pred)

            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container(border=True):
                    st.markdown(f"""<div style="margin-bottom:15px; text-align:center; font-size:25px"><span style="font-weight:bold">MAE</span><br>US$ {mae:.2f}</div>""", unsafe_allow_html=True)
            with col2:
                with st.container(border=True):
                    st.markdown(f"""<div style="margin-bottom:15px; text-align:center; font-size:25px"><span style="font-weight:bold">RMSE</span><br>US$ {rmse:.2f}</div>""", unsafe_allow_html=True)
            with col3:
                with st.container(border=True):
                    st.markdown(f"""<div style="margin-bottom:15px; text-align:center; font-size:25px"><span style="font-weight:bold">R²</span><br>{r2:.4f}</div>""", unsafe_allow_html=True)
