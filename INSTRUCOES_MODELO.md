# 📋 Instruções para Carregar o Modelo

## ⚠️ Problema: Modelo não encontrado

A API precisa do arquivo do modelo treinado para funcionar. O arquivo `final_CNN_model.h5` não foi encontrado.

## 🔧 Solução 1: Treinar o Modelo (Recomendado)

1. Abra o notebook `DeppLearning.ipynb` no Jupyter ou VS Code
2. Execute todas as células do notebook até a parte do treinamento da CNN
3. Execute a célula que salva o modelo:
   ```python
   model.save('./model/final_CNN_model.h5')
   ```
4. Certifique-se de que a pasta `model` foi criada e contém o arquivo `final_CNN_model.h5`

## 🔧 Solução 2: Baixar o Modelo Existente

Se você já treinou o modelo antes:

1. Verifique se o arquivo existe em:
   - `./model/final_CNN_model.h5`
   - `final_CNN_model.h5` (raiz do projeto)

2. Se o arquivo estiver em outro local, copie para um dos locais acima

## 📁 Estrutura Esperada

```
PP_individual/
├── api.py
├── DeppLearning.ipynb
├── model/                          # ← Criar esta pasta se não existir
│   └── final_CNN_model.h5         # ← Arquivo do modelo aqui
├── README.md
└── ...
```

## ✅ Verificar se Funcionou

1. Inicie o servidor da API:
   ```powershell
   .venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Verifique a mensagem no console:
   - ✅ `Modelo carregado com sucesso de: ./model/final_CNN_model.h5`
   - ❌ Se aparecer `⚠️ AVISO: Modelo não encontrado!`, verifique os passos acima

3. Acesse `http://localhost:8000/upload` e teste com uma imagem

## 📝 Nota

O modelo precisa ser treinado apenas uma vez. Depois de ter o arquivo `.h5`, você pode usar a API normalmente.

