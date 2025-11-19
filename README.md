# API de Reconhecimento de Dígitos MNIST

API desenvolvida com FastAPI para reconhecimento de dígitos escritos à mão usando um modelo de Rede Neural Convolucional (CNN) treinado no conjunto de dados MNIST.

## 📋 Índice

- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Rotas da API](#rotas-da-api)
- [Exemplos de Uso](#exemplos-de-uso)
- [Estrutura do Projeto](#estrutura-do-projeto)

## 🔧 Requisitos

- Python 3.8 ou superior
- TensorFlow 2.13 ou superior
- FastAPI
- Uvicorn
- PIL (Pillow)
- NumPy

## 📦 Instalação

### 1. Clone ou navegue até o diretório do projeto

```bash
cd "c:\Users\Zion & Mariana\Desktop\Atvidades\PP_individual"
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv .venv
```

### 3. Ative o ambiente virtual

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

Ou manualmente:
```bash
pip install fastapi uvicorn[standard] tensorflow pillow numpy python-multipart requests
```

### 5. Verifique se o modelo existe

Certifique-se de que o arquivo do modelo está em um dos seguintes locais:
- `./model/final_CNN_model.h5` (diretório `model`)
- `final_CNN_model.h5` (raiz do projeto)

A API tentará carregar o modelo automaticamente de um desses locais.

## 🚀 Como Executar

### Executar o servidor (PowerShell)

**Opção 1: Executar diretamente**
```powershell
cd "c:\Users\Zion & Mariana\Desktop\Atvidades\PP_individual"
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Opção 2: Usar ponto e vírgula (PowerShell)**
```powershell
cd "c:\Users\Zion & Mariana\Desktop\Atvidades\PP_individual"; uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Opção 3: Comandos separados**
```powershell
# Navegar para o diretório
cd "c:\Users\Zion & Mariana\Desktop\Atvidades\PP_individual"

# Executar o servidor
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Executar o servidor (CMD/Bash)

```bash
cd "c:\Users\Zion & Mariana\Desktop\Atvidades\PP_individual" && uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

O parâmetro `--reload` permite que o servidor reinicie automaticamente quando você fizer alterações no código.

### Acessar a API

Após iniciar o servidor, a API estará disponível em:
- **URL base:** `http://localhost:8000`
- **Documentação interativa (Swagger):** `http://localhost:8000/docs`
- **Documentação alternativa (ReDoc):** `http://localhost:8000/redoc`

## 📡 Rotas da API

### 1. GET `/`

**Descrição:** Retorna informações básicas sobre a API e os endpoints disponíveis.

**Resposta de exemplo:**
```json
{
  "message": "API funcionando!",
  "endpoints": {
    "predict": "/predict",
    "docs": "/docs",
    "health": "/health"
  },
  "model_loaded": true
}
```

**Exemplo de uso:**
```bash
curl http://localhost:8000/
```

---

### 2. GET `/health`

**Descrição:** Verifica o status de saúde da API e se o modelo foi carregado corretamente.

**Resposta de exemplo:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Exemplo de uso:**
```bash
curl http://localhost:8000/health
```

---

### 3. POST `/predict`

**Descrição:** Recebe uma imagem de um dígito escrito à mão e retorna a predição do modelo.

**Parâmetros:**
- `file` (multipart/form-data): Arquivo de imagem (PNG, JPG, JPEG, etc.)

**Resposta de sucesso:**
```json
{
  "success": true,
  "predicted_class": 5,
  "confidence": 0.9876,
  "class_probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    "2": 0.0015,
    "3": 0.0023,
    "4": 0.0008,
    "5": 0.9876,
    "6": 0.0012,
    "7": 0.0015,
    "8": 0.0021,
    "9": 0.0028
  },
  "all_probabilities": [0.0001, 0.0002, ...]
}
```

**Campos da resposta:**
- `success`: Indica se a predição foi bem-sucedida
- `predicted_class`: Dígito previsto (0-9)
- `confidence`: Nível de confiança da predição (0.0 a 1.0)
- `class_probabilities`: Probabilidade de cada classe (0-9)
- `all_probabilities`: Array com todas as probabilidades

**Códigos de erro:**
- `400`: Erro ao processar a imagem ou arquivo inválido
- `500`: Modelo não foi carregado

**Exemplo de uso com curl:**
```bash
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@sua_imagem.png"
```

**Exemplo de uso com Python:**
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("sua_imagem.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

### 4. GET `/docs`

**Descrição:** Interface Swagger UI para documentação interativa e teste da API diretamente no navegador.

**Acesso:** `http://localhost:8000/docs`

---

### 5. GET `/redoc`

**Descrição:** Documentação alternativa em formato ReDoc.

**Acesso:** `http://localhost:8000/redoc`

## 💡 Exemplos de Uso

### Testar a API usando Python

```python
import requests

# URL da API
API_URL = "http://localhost:8000"

# 1. Verificar status
response = requests.get(f"{API_URL}/health")
print("Status:", response.json())

# 2. Fazer predição
with open("imagem_digito.png", "rb") as f:
    files = {"file": ("imagem.png", f, "image/png")}
    response = requests.post(f"{API_URL}/predict", files=files)
    
result = response.json()
print(f"Dígito previsto: {result['predicted_class']}")
print(f"Confiança: {result['confidence']:.2%}")
```

### Testar usando o script fornecido

Execute o script de teste incluído no projeto:

```bash
python test_api.py
```

### Testar usando a interface Swagger

1. Inicie o servidor
2. Acesse `http://localhost:8000/docs`
3. Clique em `/predict` → "Try it out"
4. Faça upload de uma imagem
5. Clique em "Execute" para ver o resultado

## 📝 Notas Importantes

### Processamento de Imagens

- A API aceita imagens em qualquer tamanho e formato (PNG, JPG, JPEG, etc.)
- As imagens são automaticamente convertidas para escala de cinza
- As imagens são redimensionadas para 28x28 pixels (formato MNIST)
- Se a imagem tiver fundo branco, a API inverte automaticamente as cores (MNIST usa fundo preto e dígito branco)

### Requisitos do Modelo

- O modelo espera imagens de 28x28 pixels em escala de cinza
- As imagens devem ser normalizadas (valores entre 0 e 1)
- O formato de entrada é (1, 28, 28, 1) - (batch, altura, largura, canais)

### CORS

A API está configurada para aceitar requisições de qualquer origem. Isso é útil para desenvolvimento, mas pode ser restrito em produção alterando a configuração do `CORSMiddleware` no arquivo `api.py`.

## 📁 Estrutura do Projeto

```
PP_individual/
├── api.py                 # Arquivo principal da API FastAPI
├── test_api.py            # Script de teste da API
├── DeppLearning.ipynb     # Notebook com treinamento dos modelos
├── requirements.txt       # Dependências do projeto
├── README.md              # Este arquivo
├── .venv/                 # Ambiente virtual (se criado)
└── model/                 # Diretório para o modelo (se existir)
    └── final_CNN_model.h5 # Modelo CNN treinado
```

## 🐛 Solução de Problemas

### Erro: "Modelo não foi carregado"

- Verifique se o arquivo `final_CNN_model.h5` existe em `./model/` ou na raiz do projeto
- Verifique se há erros no console ao iniciar o servidor

### Erro: "Erro ao processar imagem"

- Certifique-se de que o arquivo enviado é uma imagem válida
- Verifique se o formato da imagem é suportado (PNG, JPG, JPEG, etc.)

### Porta 8000 já está em uso

Use uma porta diferente:
```bash
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

### Erro com `&&` no PowerShell

No PowerShell do Windows, use `;` ao invés de `&&`:

```powershell
cd "caminho"; uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Ou execute os comandos separadamente.

## 📚 Tecnologias Utilizadas

- **FastAPI**: Framework web moderno e rápido para Python
- **TensorFlow/Keras**: Framework de deep learning para carregar e usar o modelo
- **Uvicorn**: Servidor ASGI de alta performance
- **PIL (Pillow)**: Processamento de imagens
- **NumPy**: Operações numéricas e manipulação de arrays

## 📄 Licença

Este projeto é parte de um trabalho individual de Processamento de Padrões.

