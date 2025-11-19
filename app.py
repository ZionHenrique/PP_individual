from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow import keras
import os

app = Flask(__name__)

# Carregar modelo MNIST
modelo = keras.models.load_model("modelo_mnist.h5")

# Pasta para uploads
offset = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(offset, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


def preprocessar_imagem(caminho):
    """Preprocessa a imagem para o formato MNIST (28x28, grayscale, normalizado)"""
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

    # Reduz ruído e aumenta contraste
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Resize para 28x28
    img = cv2.resize(img, (28, 28))

    # Normaliza (0–1)
    img = img.astype("float32") / 255.0

    # Flatten (784 posições)
    img = img.reshape(1, 28 * 28)

    return img


@app.route('/upload', methods=['POST'])
def upload():
    if 'imagem' not in request.files:
        return "Nenhuma imagem enviada"

    file = request.files['imagem']

    if file.filename == '':
        return "Arquivo inválido"

    # Salvar imagem
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    try:
        entrada = preprocessar_imagem(img_path)

        # Predição
        pred = modelo.predict(entrada)
        numero = int(np.argmax(pred))  # dígito reconhecido

    except Exception as e:
        return f"Erro ao processar: {e}"

    return render_template("resultado.html", texto=str(numero))


@app.route('/resultado')
def resultado():
    texto = request.args.get('texto', '')
    return render_template('resultado.html', texto=texto)


if __name__ == '__main__':
    app.run(debug=True)
