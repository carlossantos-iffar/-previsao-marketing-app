import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo, os encoders e a lista de colunas
try:
    model = joblib.load('modelo_random_forest.joblib')
    encoders = joblib.load('encoders.joblib')
    model_columns = joblib.load('colunas_modelo.joblib')
except FileNotFoundError:
    st.error("Arquivos de modelo ('modelo_random_forest.joblib') ou encoders ('encoders.joblib') não encontrados. Certifique-se de que eles estão na mesma pasta que o app.py.")
    st.stop()


# --- FUNÇÃO DE PRÉ-PROCESSAMENTO ---
# Esta função pega os dados do usuário e os prepara para o modelo
def preprocess_input(data):
    # Cria um DataFrame com os dados do usuário
    user_df = pd.DataFrame([data])

    # Aplica os encoders nas colunas categóricas
    for column, encoder in encoders.items():
        # Verificamos se a coluna existe no DataFrame do usuário
        if column in user_df.columns and column != 'y': # Não processamos a coluna alvo 'y'
            # Pega o valor que o usuário inseriu
            valor_usuario = user_df[column].iloc[0]
            # Transforma o valor usando o encoder carregado
            user_df[column] = encoder.transform([valor_usuario])[0]

    # Garante que a ordem das colunas está correta
    # Cria um dataframe com zeros e as colunas corretas
    processed_df = pd.DataFrame(columns=model_columns)
    # Adiciona uma linha de zeros
    processed_df.loc[0] = 0
    # Preenche com os dados do usuário
    for col in user_df.columns:
        if col in processed_df.columns:
            processed_df[col] = user_df[col].values[0]

    return processed_df


# --- INTERFACE DO USUÁRIO COM STREAMLIT ---

st.set_page_config(page_title="Previsão de Marketing Bancário", layout="wide")
st.title('🤖 Aplicação de Previsão de Adesão a Campanha Bancária')
st.write('Esta aplicação utiliza um modelo de Machine Learning (Random Forest) para prever se um cliente irá aderir a uma campanha de marketing de um banco.')
st.write('Preencha os dados do cliente abaixo para receber a previsão.')

st.divider()

# Criando colunas para organizar os campos de entrada
col1, col2, col3 = st.columns(3)

# Dicionário para armazenar as entradas do usuário
user_input = {}

with col1:
    st.subheader("Informações Pessoais")
    user_input['age'] = st.number_input('Idade', min_value=18, max_value=100, value=40)
    # Usamos as classes salvas no encoder para criar as opções do selectbox
    user_input['job'] = st.selectbox('Profissão', options=encoders['job'].classes_)
    user_input['marital'] = st.selectbox('Estado Civil', options=encoders['marital'].classes_)
    user_input['education'] = st.selectbox('Escolaridade', options=encoders['education'].classes_)

with col2:
    st.subheader("Histórico com o Banco")
    user_input['default'] = st.selectbox('Possui Inadimplência?', options=encoders['default'].classes_)
    user_input['balance'] = st.number_input('Saldo Médio Anual (em Euros)', value=1500)
    user_input['housing'] = st.selectbox('Possui Empréstimo Imobiliário?', options=encoders['housing'].classes_)
    user_input['loan'] = st.selectbox('Possui Empréstimo Pessoal?', options=encoders['loan'].classes_)

with col3:
    st.subheader("Última Campanha")
    user_input['contact'] = st.selectbox('Meio de Contato', options=encoders['contact'].classes_)
    user_input['day'] = st.slider('Último Dia de Contato', 1, 31, 15)
    user_input['month'] = st.selectbox('Último Mês de Contato', options=encoders['month'].classes_)
    user_input['duration'] = st.number_input('Duração do Último Contato (segundos)', value=200, min_value=0)
    user_input['campaign'] = st.number_input('Nº de Contatos Nesta Campanha', value=1, min_value=1)
    user_input['pdays'] = st.number_input('Dias Desde o Último Contato (campanha anterior)', value=-1, min_value=-1)
    user_input['previous'] = st.number_input('Nº de Contatos (campanha anterior)', value=0, min_value=0)
    user_input['poutcome'] = st.selectbox('Resultado da Campanha Anterior', options=encoders['poutcome'].classes_)


# Botão de Previsão
if st.button('Fazer Previsão', type="primary"):
    # 1. Pré-processar os dados do usuário
    processed_data = preprocess_input(user_input)

    # 2. Fazer a previsão com o modelo
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)

    # 3. Mostrar o resultado
    st.divider()
    st.subheader('Resultado da Previsão:')

    # O resultado da previsão será 0 ou 1.
    # Usamos o encoder de 'y' para traduzir de volta para 'yes' ou 'no'.
    resultado_texto = encoders['y'].inverse_transform(prediction)[0]

    if resultado_texto == 'yes':
        st.success('O cliente provavelmente VAI ADERIR à campanha! ✅')
        st.write(f"**Confiança da Previsão:** {prediction_proba[0][1]*100:.2f}%")
    else:
        st.error('O cliente provavelmente NÃO VAI ADERIR à campanha. ❌')
        st.write(f"**Confiança da Previsão:** {prediction_proba[0][0]*100:.2f}%")