import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pickle import load
import os

# Title of the app
st.title("Probilidade de um aluno do Passos Mágicos abandonar uma turma")

# Load your data
@st.cache_data
def load_data():
    # Replace this with the path to your data
    data = pd.read_csv('previsões.csv')
    return data

# Load the data
data = load_data()

# # Display the dataframe
# st.subheader("Data")
# st.dataframe(data)

# Visualização de disbribuição de alunos nas turmas
st.subheader("Selecione um corte de Probabilidade para observar os alunos com chance de Abandonar")
corte = st.slider(label='Probabilidade (%) de Abandonar a turma', min_value=0.0, max_value=100.0)
data['Classe'] = np.where((data['ProbabilidadeDesistente']*100) >= corte, "Abandono", "Não Abandono")
df_counts = data['Classe'].value_counts(dropna=False).to_frame().rename(columns={'count':'QtdAlunos'})
df_perc_counts = data['Classe'].value_counts(dropna=False, normalize=True).to_frame().rename(columns={'proportion':'%QtdAlunos'})
df_perc_counts['%QtdAlunos'] = round(df_perc_counts['%QtdAlunos'] * 100, 2) 
st.dataframe(df_counts.join(df_perc_counts))

# Select student ID or name to predict
st.subheader("Selecione um aluno para ver a sua probabilidade de abandono")
student_id = st.selectbox("Escolha um IdAluno", data['IdAluno'].unique())

if student_id:
    st.subheader("Selecione uma turma para o aluno")
    class_id = st.selectbox("Escolha um IdTurma", data[data['IdAluno'] == student_id]['IdTurma'].unique())

# Display selected student's prediction details
student_data = data[(data['IdAluno'] == student_id) & (data['IdTurma'] == class_id)]

if not student_data.empty:
    st.write(f"Previsão para o IdALuno {student_id}")
    st.write(f"Probabilidade de Abandonar a turma: {student_data['ProbabilidadeDesistente'].values[0]:.2%}")
    st.write(f"Previsão: {student_data['Classe'].values[0]}")

    # Plot the probability
    fig, ax = plt.subplots()
    ax.barh([student_id], [student_data['ProbabilidadeDesistente'].values[0]], color='blue')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilidade de Abandonar')
    st.pyplot(fig)

# Option to show all predictions
st.subheader("Previsões para todos os alunos")
if st.checkbox("Ver todas as previsões"):
    st.write(data)

# Download the prediction results
st.subheader("Download dos Resultados")
csv = data.to_csv(index=False)
st.download_button(label="Download CSV", data=csv, file_name='student_churn_predictions.csv', mime='text/csv')

st.subheader("Upload de alunos para previsão")
uploaded_file = st.file_uploader(
    "Escolha o arquivo", accept_multiple_files=False
)

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    # Carregar o modelo
    try:
        model = load(open('models/best_model_XGBoost_2024-08-16.pkl', 'rb'))
    except:
        model = load(open(os.getcwd()+'/models/best_model_XGBoost_2024-08-16.pkl', 'rb'))

    # Lista de variáveis a serem desconsideradas 
    lst_outras_variaveis = ['IdAluno','IdTurma','IdPeriodo','IdSerie','AnoMesRefFeatures','AnoRefFeatures','DataRefFeaturesUltDia','TargetDesistente']

    # Separando as features entre numericas e categoricas
    lst_features_categories = ['TipoTurma', 'TurnoPrincipal', 'IdDisciplina',
        'DisciplinaRequerNota', 
        'DisciplinaStUsaNotaConceito', 'SexoTurma', 'IdProfessor', 'Sexo',
        'EstadoCivil', 'CorRaca', 'StDeficienciaCegueira',
        'StDeficienciaBaixaVisao', 'StDeficienciaSurdez',
        'StDeficienciaAuditiva', 'StDeficienciaFisica',
        'StDeficienciaSurdoCegueira', 'StDeficienciaMultipla',
        'StDeficienciaMental', 'StDeficienciaAutismoInfantil',
        'StDeficienciaSindromeAsperger', 'StDeficienciaSindromeRett',
        'StDeficienciaTrastornoDesintegrativo', 'StDeficienciaAltasHabilidades',
        'StAppComunicacao_Sincronizado', 'StRecursoAuxilioLedor',
        'StRecursoAuxilioTranscricao', 'StRecursoGuiaInterprete',
        'StRecursoTradutorInterpreteDeLibras', 'StRecursoLeituraLabial',
        'StRecursoMaterialDidaticoProvaBraille', 'StRecursoProvaAmpliada',
        'StRecursoProvaSuperampliada', 'StRecursoCdComAudio',
        'StRecursoLinguaPortuguesaSegundaLingua', 'StRecursoProvaEmVideo',
        'StPermiteUsoImagem', 'StEstrangeiro', 'TipoResponsavel',
        'GrauParentescoResponsavel']

    lst_features_numerics = [col for col in df_input.columns if col not in lst_features_categories+lst_outras_variaveis]

    # Selecionando variáveis para a aplicação do modelo
    df_X = df_input[lst_features_categories+lst_features_numerics]

    # Previsões
    df_input['ProbabilidadeAbandono'] = model.predict_proba(df_X)[:, 1]
    df_input['Classe'] = np.where((df_input['ProbabilidadeAbandono']*100) >= corte, "Abandono", "Não Abandono")

    # 
    st.write(df_input)


# bytes_data = uploaded_file.read()
# st.text(f"{(type(bytes_data))}")
