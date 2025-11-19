from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
from tensorflow.keras.activations import softmax

# Inicia API
app = FastAPI(title="MNIST Digit Recognition API")

# Habilitar CORS para permitir requisições de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Carregar o modelo
# -------------------------------
MODEL_PATH = "./model/final_CNN_model.h5"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "final_CNN_model.h5"

try:
    # Carregar modelo com custom_objects para softmax_v2 se necessário
    custom_objects = {'softmax_v2': softmax}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Modelo carregado com sucesso de: {MODEL_PATH}")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    model = None

# Tamanho correto da imagem MNIST: 28x28
IMG_SIZE = 28

@app.get("/")
def root():
    return {
        "message": "API funcionando!",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health",
            "upload_page": "/upload"
        },
        "model_loaded": model is not None
    }

@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    """Página HTML para fazer upload de imagens"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MNIST Digit Recognition - Upload</title>
        <meta charset="UTF-8">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                padding: 40px;
                max-width: 600px;
                width: 100%;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                text-align: center;
            }
            .subtitle {
                color: #666;
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin-bottom: 20px;
                transition: all 0.3s;
                cursor: pointer;
                background: #f8f9ff;
            }
            .upload-area:hover {
                border-color: #764ba2;
                background: #f0f2ff;
            }
            .upload-area.dragover {
                border-color: #764ba2;
                background: #e8ebff;
                transform: scale(1.02);
            }
            input[type="file"] {
                display: none;
            }
            .file-label {
                display: block;
                cursor: pointer;
                color: #667eea;
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 10px;
            }
            .file-info {
                color: #999;
                font-size: 14px;
                margin-top: 10px;
            }
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s;
                margin-top: 20px;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .preview {
                margin: 20px 0;
                text-align: center;
            }
            .preview img {
                max-width: 200px;
                max-height: 200px;
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 10px;
                background: #f9f9f9;
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 10px;
                display: none;
            }
            .result.success {
                background: #d4edda;
                border: 2px solid #28a745;
                color: #155724;
            }
            .result.error {
                background: #f8d7da;
                border: 2px solid #dc3545;
                color: #721c24;
            }
            .result h2 {
                margin-bottom: 15px;
                font-size: 24px;
            }
            .result p {
                margin: 5px 0;
                font-size: 16px;
            }
            .confidence-bar {
                width: 100%;
                height: 30px;
                background: #e0e0e0;
                border-radius: 15px;
                margin: 10px 0;
                overflow: hidden;
            }
            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, #28a745, #20c997);
                transition: width 0.5s;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔢 MNIST Digit Recognition</h1>
            <p class="subtitle">Envie uma imagem de um dígito escrito à mão (0-9)</p>
            
            <div class="upload-area" id="uploadArea">
                <label for="fileInput" class="file-label">📁 Clique para selecionar ou arraste uma imagem</label>
                <input type="file" id="fileInput" accept="image/*">
                <div class="file-info">Formatos aceitos: PNG, JPG, JPEG</div>
            </div>
            
            <div class="preview" id="preview"></div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processando imagem...</p>
            </div>
            
            <button id="submitBtn" onclick="uploadImage()" disabled>🔍 Reconhecer Dígito</button>
            
            <div class="result" id="result"></div>
        </div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const uploadArea = document.getElementById('uploadArea');
            const preview = document.getElementById('preview');
            const submitBtn = document.getElementById('submitBtn');
            const result = document.getElementById('result');
            const loading = document.getElementById('loading');
            let selectedFile = null;

            // Upload area click
            uploadArea.addEventListener('click', () => fileInput.click());

            // File input change
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFile(e.target.files[0]);
                }
            });

            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) {
                    handleFile(e.dataTransfer.files[0]);
                }
            });

            function handleFile(file) {
                if (!file.type.startsWith('image/')) {
                    alert('Por favor, selecione um arquivo de imagem!');
                    return;
                }

                selectedFile = file;
                submitBtn.disabled = false;

                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                };
                reader.readAsDataURL(file);
            }

            async function uploadImage() {
                if (!selectedFile) return;

                submitBtn.disabled = true;
                loading.style.display = 'block';
                result.style.display = 'none';

                const formData = new FormData();
                formData.append('file', selectedFile);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        result.className = 'result success';
                        result.innerHTML = `
                            <h2>✅ Resultado</h2>
                            <p><strong>Dígito Previsto:</strong> ${data.predicted_class}</p>
                            <p><strong>Confiança:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${data.confidence * 100}%">
                                    ${(data.confidence * 100).toFixed(1)}%
                                </div>
                            </div>
                        `;
                    } else {
                        result.className = 'result error';
                        result.innerHTML = `
                            <h2>❌ Erro</h2>
                            <p>${data.detail || 'Erro ao processar imagem'}</p>
                        `;
                    }
                } catch (error) {
                    result.className = 'result error';
                    result.innerHTML = `
                        <h2>❌ Erro</h2>
                        <p>Erro ao conectar com a API: ${error.message}</p>
                    `;
                } finally {
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    submitBtn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não foi carregado")
    
    # Validar tipo de arquivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
    
    try:
        # Ler imagem enviada
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Converter para grayscale (L = luminância, escala de cinza)
        image = image.convert("L")

        # Redimensionar para entrada do modelo MNIST (28x28)
        image = image.resize((IMG_SIZE, IMG_SIZE))

        # Converter para array numpy e normalizar para [0, 1]
        img_array = np.array(image).astype("float32") / 255.0

        # Inverter cores se necessário (MNIST tem fundo preto e dígito branco)
        # Se a imagem tiver fundo branco, inverter
        if img_array.mean() > 0.5:
            img_array = 1.0 - img_array

        # Adicionar dimensões: batch e canal (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=0)  # Adiciona batch
        img_array = np.expand_dims(img_array, axis=-1)  # Adiciona canal

        # Predição
        preds = model.predict(img_array, verbose=0)
        preds_list = preds[0].tolist()

        # Classe com maior probabilidade
        predicted_class = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # Criar dicionário com probabilidades de cada classe
        class_probabilities = {
            str(i): round(prob, 4) 
            for i, prob in enumerate(preds_list)
        }

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probabilities,
            "all_probabilities": preds_list
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

