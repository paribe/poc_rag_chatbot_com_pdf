# Chatbot com IA usando PDFs e GROQ

Este projeto é um chatbot que permite fazer perguntas sobre documentos PDF usando inteligência artificial. O chatbot utiliza o modelo GROQ para processar as perguntas e fornecer respostas baseadas no conteúdo dos documentos.

## 🚀 Funcionalidades

- Upload de múltiplos arquivos PDF
- Processamento automático dos documentos
- Interface de chat amigável
- Suporte a diferentes modelos do GROQ
- Respostas baseadas no contexto dos documentos

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Conta no GROQ (para obter a chave da API)
- Pip (gerenciador de pacotes Python)

## 🔧 Instalação

1. Clone este repositório ou baixe os arquivos

2. Crie um ambiente virtual Python (recomendado):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Crie um arquivo `.env` na pasta do projeto com sua chave da API do GROQ:
```
GROQ_API_KEY=sua_chave_groq_aqui
```

## 🎮 Como Usar

1. Inicie o aplicativo:
```bash
streamlit run app.py
```

2. No navegador, você verá a interface do chatbot com:
   - Uma barra lateral para upload de arquivos PDF
   - Um seletor de modelo do GROQ
   - Uma área de chat para fazer perguntas

3. Para começar:
   - Faça upload de um ou mais arquivos PDF usando o botão na barra lateral
   - Selecione o modelo do GROQ que deseja usar
   - Digite sua pergunta na caixa de chat
   - Aguarde a resposta do chatbot

## 🤖 Modelos Disponíveis

O chatbot suporta os seguintes modelos do GROQ:
- mixtral-8x7b-32768
- llama2-70b-4096
- gemma-7b-it

## 📝 Exemplo de Uso

1. Faça upload de um manual técnico em PDF
2. Selecione o modelo "mixtral-8x7b-32768"
3. Faça perguntas como:
   - "Qual é o procedimento para ligar o equipamento?"
   - "Quais são as especificações técnicas?"
   - "Como faço a manutenção preventiva?"

## 🔍 Como Funciona

1. **Upload de Documentos**:
   - Os PDFs são carregados e processados
   - O texto é dividido em partes menores (chunks)
   - As partes são armazenadas em um banco de dados vetorial

2. **Processamento de Perguntas**:
   - Sua pergunta é enviada para o modelo GROQ
   - O sistema busca as partes relevantes dos documentos
   - O modelo gera uma resposta baseada no contexto encontrado

## ⚠️ Limitações

- O tamanho máximo dos PDFs pode variar dependendo da memória disponível
- A qualidade das respostas depende do modelo escolhido
- O processamento de PDFs muito grandes pode levar mais tempo

## 🛠️ Tecnologias Utilizadas

- Python
- Streamlit
- LangChain
- GROQ API
- ChromaDB (banco de dados vetorial)

## 📚 Recursos Adicionais

- [Documentação do GROQ](https://console.groq.com/docs)
- [Documentação do Streamlit](https://docs.streamlit.io/)
- [Documentação do LangChain](https://python.langchain.com/docs/get_started/introduction)

## 🤝 Contribuindo

Sinta-se à vontade para contribuir com o projeto:
1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request


