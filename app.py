import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo, os encoders, as colunas e o scaler
try:
    model = joblib.load('modelo_svm_model.joblib')
    encoders = joblib.load('encoders.joblib')
    model_columns = joblib.load('colunas_modelo.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Arquivos de modelo, encoders ou scaler não encontrados. Certifique-se de que estão na mesma pasta que o app.py.")
    st.stop()

# --- FUNÇÃO DE PRÉ-PROCESSAMENTO ---
def preprocess_input(data):
    user_df = pd.DataFrame([data])

    # Aplica os encoders nas colunas categóricas
    for column, encoder in encoders.items():
        if column in user_df.columns and column != 'y':
            valor_usuario = user_df[column].iloc[0]
            if valor_usuario in encoder.classes_:
                user_df[column] = encoder.transform([valor_usuario])[0]
            else:
                st.error(f"Valor '{valor_usuario}' não encontrado no encoder da coluna '{column}'.")
                st.stop()

    # Garante a estrutura correta das colunas
    processed_df = pd.DataFrame(columns=model_columns)
    processed_df.loc[0] = 0  # Inicializa todas as colunas com zero

    for col in user_df.columns:
        if col in processed_df.columns:
            processed_df[col] = user_df[col].values[0]

    # Aplica o scaler para normalizar os dados
    processed_scaled = scaler.transform(processed_df)

    return processed_scaled

# --- INTERFACE DO USUÁRIO COM STREAMLIT ---

st.set_page_config(page_title="Previsão de Marketing Bancário", layout="wide")
st.title('🤖 Aplicação de Previsão de Adesão a uma Campanha Bancária')
st.title('Por: Professor Carlos Santos - Instituto Federal Farroupilha Câmpus Alegrete')
st.write('Esta aplicação utiliza um modelo de Machine Learning (SVM) para prever se um cliente irá aderir a uma campanha de marketing de um banco.')
st.write('Preencha os dados do cliente abaixo para receber a previsão.')

st.divider()

col1, col2, col3 = st.columns(3)

user_input = {}

with col1:
    st.subheader("Informações Pessoais")
    user_input['age'] = st.number_input('Idade', min_value=18, max_value=100, value=40)
    user_input['job'] = st.selectbox('Profissão', options=encoders['job'].classes_)
    user_input['marital'] = st.selectbox('Estado Civil', options=encoders['marital'].classes_)
    user_input['education'] = st.selectbox('Escolaridade', options=encoders['education'].classes_)

with col2:
    st.subheader("Histórico com o Banco")
    user_input['default'] = st.selectbox('Possui Inadimplência?', options=encoders['default'].classes_)
    user_input['housing'] = st.selectbox('Possui Empréstimo Imobiliário?', options=encoders['housing'].classes_)
    user_input['loan'] = st.selectbox('Possui Empréstimo Pessoal?', options=encoders['loan'].classes_)

with col3:
    st.subheader("Última Campanha")
    user_input['contact'] = st.selectbox('Meio de Contato', options=encoders['contact'].classes_)
    user_input['month'] = st.selectbox('Último Mês de Contato', options=encoders['month'].classes_)
    user_input['day_of_week'] = st.selectbox('Dia da Semana do Contato', options=encoders['day_of_week'].classes_)
    user_input['duration'] = st.number_input('Duração do Último Contato (segundos)', value=200, min_value=0)
    user_input['campaign'] = st.number_input('Nº de Contatos Nesta Campanha', value=1, min_value=1)
    user_input['pdays'] = st.number_input('Dias desde Último Contato (campanha anterior)', value=-1, min_value=-1)
    user_input['previous'] = st.number_input('Nº de Contatos (campanha anterior)', value=0, min_value=0)
    user_input['poutcome'] = st.selectbox('Resultado da Campanha Anterior', options=encoders['poutcome'].classes_)

# Botão de Previsão
if st.button('Fazer Previsão', type="primary"):
    processed_data = preprocess_input(user_input)
    prediction_proba = model.predict_proba(processed_data)[0][1]
    threshold = 0.3
    prediction = int(prediction_proba >= threshold)

    st.divider()
    st.subheader('Resultado da Previsão:')

    resultado_texto = encoders['y'].inverse_transform([prediction])[0]

    if resultado_texto == 'yes':
        st.success('O cliente provavelmente VAI ADERIR à campanha! ✅')
        st.write(f"**Confiança da Previsão:** {prediction_proba*100:.2f}%")
    else:
        st.error('O cliente provavelmente NÃO VAI ADERIR à campanha. ❌')
        st.write(f"**Confiança da Previsão:** {(1 - prediction_proba)*100:.2f}%")
