import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo, os encoders, as colunas e o scaler
try:
    model = joblib.load('modelo_random_forest.joblib')
    encoders = joblib.load('encoders.joblib')
    model_columns = joblib.load('colunas_modelo.joblib')
    scaler = joblib.load('scaler.joblib')  # ✅ carregando o scaler
except FileNotFoundError:
    st.error("Arquivos de modelo, encoders ou scaler não encontrados. Certifique-se de que estão na mesma pasta que o app.py.")
    st.stop()

# --- FUNÇÃO DE PRÉ-PROCESSAMENTO ---
def preprocess_input(data):
    # Cria um DataFrame com os dados do usuário
    user_df = pd.DataFrame([data])

    # Aplica os encoders nas colunas categóricas
    for column, encoder in encoders.items():
        if column in user_df.columns and column != 'assinou_deposito':
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

    # ✅ Aplica o scaler para normalizar os dados
    processed_scaled = scaler.transform(processed_df)

    return processed_scaled

# --- INTERFACE DO USUÁRIO COM STREAMLIT ---

st.set_page_config(page_title="Previsão de Marketing Bancário", layout="wide")
st.title('🤖 Aplicação de Previsão de Adesão a uma Campanha Bancária')
st.title('Por: Professor Carlos Santos - Instituto Federal Farroupilha Câmpus Alegrete')
st.write('Esta aplicação utiliza um modelo de Machine Learning (Random Forest) para prever se um cliente irá aderir a uma campanha de marketing de um banco.')
st.write('Preencha os dados do cliente abaixo para receber a previsão.')

st.divider()

# Criando colunas para organizar os campos de entrada
col1, col2, col3 = st.columns(3)

# Dicionário para armazenar as entradas do usuário
user_input = {}

with col1:
    st.subheader("Informações Pessoais")
    user_input['idade'] = st.number_input('Idade', min_value=18, max_value=100, value=40)
    user_input['profissao'] = st.selectbox('Profissão', options=encoders['profissao'].classes_)
    user_input['estado_civil'] = st.selectbox('Estado Civil', options=encoders['estado_civil'].classes_)
    user_input['escolaridade'] = st.selectbox('Escolaridade', options=encoders['escolaridade'].classes_)

with col2:
    st.subheader("Histórico com o Banco")
    user_input['inadimplente'] = st.selectbox('Possui Inadimplência?', options=encoders['inadimplente'].classes_)
    user_input['saldo_medio_anual'] = st.number_input('Saldo Médio Anual (em Euros)', value=1500)
    user_input['emprestimo_habitacional'] = st.selectbox('Possui Empréstimo Imobiliário?', options=encoders['emprestimo_habitacional'].classes_)
    user_input['emprestimo_pessoal'] = st.selectbox('Possui Empréstimo Pessoal?', options=encoders['emprestimo_pessoal'].classes_)

with col3:
    st.subheader("Última Campanha")
    user_input['tipo_contato'] = st.selectbox('Meio de Contato', options=encoders['tipo_contato'].classes_)
    user_input['dia_da_semana'] = st.slider('Último Dia da Semana de Contato', 0, 6, 3)
    user_input['mes_contato'] = st.selectbox('Último Mês de Contato', options=encoders['mes_contato'].classes_)
    user_input['duracao_contato'] = st.number_input('Duração do Último Contato (segundos)', value=200, min_value=0)
    user_input['numero_contatos'] = st.number_input('Nº de Contatos Nesta Campanha', value=1, min_value=1)
    user_input['dias_ultimo_contato'] = st.number_input('Dias Desde o Último Contato (campanha anterior)', value=-1, min_value=-1)
    user_input['contatos_anteriores'] = st.number_input('Nº de Contatos (campanha anterior)', value=0, min_value=0)
    user_input['resultado_campanha_anterior'] = st.selectbox('Resultado da Campanha Anterior', options=encoders['resultado_campanha_anterior'].classes_)

# Botão de Previsão
if st.button('Fazer Previsão', type="primary"):
    processed_data = preprocess_input(user_input)
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)

    st.divider()
    st.subheader('Resultado da Previsão:')

    resultado_texto = encoders['assinou_deposito'].inverse_transform(prediction)[0]

    if resultado_texto == 'yes':
        st.success('O cliente provavelmente VAI ADERIR à campanha! ✅')
        st.write(f"**Confiança da Previsão:** {prediction_proba[0][1]*100:.2f}%")
    else:
        st.error('O cliente provavelmente NÃO VAI ADERIR à campanha. ❌')
        st.write(f"**Confiança da Previsão:** {prediction_proba[0][0]*100:.2f}%")