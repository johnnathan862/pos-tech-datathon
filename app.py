import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

# Display the dataframe
st.subheader("Data")
st.dataframe(data)

# Select student ID or name to predict
st.subheader("Selecione um aluno para ver a sua probabilidade de abandono")
student_id = st.selectbox("Escolha um IdAluno", data['IdAluno'].unique())

# Display selected student's prediction details
student_data = data[data['IdAluno'] == student_id]

if not student_data.empty:
    st.write(f"Previsão para o IdALuno {student_id}")
    st.write(f"Probabilidade de Abandonar a turma: {student_data['ProbabilidadeDesistente'].values[0]:.2%}")
    st.write(f"Previsão: {'Abandono' if student_data['ProbabilidadeDesistente'].values[0] >= 0.5 else 'Não Abandono'}")

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
