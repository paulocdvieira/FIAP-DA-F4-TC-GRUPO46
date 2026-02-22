# FIAP-DA-F4-TC-GRUPO46

⚖️ Modelo Preditivo de Diagnóstico de Obesidade
Este repositório contém a solução desenvolvida para o Tech Challenge, focada na criação de um modelo de Machine Learning capaz de classificar níveis de obesidade com base em dados biométricos e hábitos de vida.

🚀 Estrutura do Projeto
O projeto está dividido em duas frentes principais:

Desenvolvimento do Modelo (.ipynb / .joblib): Onde a inteligência foi treinada.

Interface de Usuário (app.py): Onde o modelo é consumido via Streamlit.

🧠 Parte 1: O Modelo Preditivo
O modelo foi treinado utilizando o algoritmo Random Forest Classifier, escolhido por sua alta capacidade de lidar com variáveis categóricas e fornecer a importância de cada fator no diagnóstico.

Público-alvo: Indivíduos entre 14 e 61 anos.

Variáveis Analisadas:

Biometria: Idade, Altura, Peso e IMC (calculado).

Hábitos Alimentares: Consumo de vegetais, água e lanches calóricos.

Estilo de Vida: Frequência de atividade física, tabagismo, consumo de álcool e tempo de uso de eletrônicos.

Técnicas Utilizadas: * Pipeline de processamento customizado.

Codificação Ordinal e One-Hot Encoding.

Normalização com Min-Max Scaler.

💻 Parte 2: Interface Streamlit (app.py)
Para tornar o modelo acessível, criamos uma aplicação web interativa que permite realizar diagnósticos em tempo real.

✨ Funcionalidades
Formulário Inteligente: Campos intuitivos com legendas de ajuda (help) para facilitar o preenchimento.

Visualização de Resultados: * Diagnóstico direto conforme a classificação da OMS.

Gráfico de barras indicando a posição do usuário na escala de IMC.

Explicabilidade (XAI): Um gráfico de importância das variáveis mostra exatamente quais fatores (ex: peso, sedentarismo) mais influenciaram o resultado.

Exportação: Opção de baixar o relatório do diagnóstico em formato de texto.

Gestão de Sessão: Botão de "Nova Consulta" que limpa todos os dados de forma instantânea através de chaves dinâmicas (Session State).

🛠️ Como rodar o projeto localmente
Clone o repositório:

Bash
git clone https://github.com/seu-usuario/seu-repositorio.git
Instale as dependências:

Bash
pip install -r requirements.txt
Execute a aplicação:

Bash
streamlit run app.py
🛠️ Tecnologias Utilizadas
Python 3.10+

Scikit-Learn: Criação e treinamento do modelo.

Pandas & Numpy: Manipulação de dados.

Matplotlib & Seaborn: Visualização de gráficos.

Streamlit: Framework para a aplicação web.

Joblib: Persistência do modelo treinado.
