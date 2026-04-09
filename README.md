# 🔬 Project Analyzer v2.0

Analisador de Filme Radiocrômico EBT3 baseado em técnicas do **Dosepy** (Luis Olivares) e **OMG Dosimetry** (Jean-Francois Cabana).

## 🚀 Deploy no Streamlit Cloud

### Passo 1: Criar conta no GitHub
- Acesse: https://github.com/signup
- Crie uma conta gratuita

### Passo 2: Criar Repositório
1. Clique no **+** → **New repository**
2. Nome: `project-analyzer`
3. Deixe **Public**
4. Clique **Create repository**

### Passo 3: Fazer Upload dos Arquivos
No repositório criado:
1. Clique em **"Add file"** → **"Upload files"**
2. Faça upload destes 3 arquivos:
   - `project_analyzer_v2.py`
   - `requirements.txt`
   - `README.md`
3. Clique **"Commit changes"**

### Passo 4: Deploy no Streamlit Cloud
1. Acesse: https://share.streamlit.io
2. Clique **"Continue with GitHub"**
3. Clique **"New app"**
4. Selecione:
   - **Repository**: `seu-usuario/project-analyzer`
   - **Branch**: `main`
   - **Main file path**: `project_analyzer_v2.py`
5. Clique **"Deploy!"**

## 📋 Funcionalidades

- ✅ Detecção automática do filme (Otsu + morfologia)
- ✅ Detecção de regiões irradiadas (Multi-Otsu)
- ✅ Ordenação por escurecimento (1=claro → N=escuro)
- ✅ Contornos e numeração automática
- ✅ Exportação CSV e PNG
- ✅ Configuração de DPI

## 🧪 Tecnologias

- **scikit-image**: Detecção e segmentação
- **Streamlit**: Interface web
- **Pandas**: Análise de dados
- **PIL**: Processamento de imagens

## 📚 Referências

- Dosepy: https://github.com/LuisOlivaresJ/Dosepy
- OMG Dosimetry: https://github.com/jfcabana/omg_dosimetry
