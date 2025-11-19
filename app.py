from flask import Flask, render_template, request
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Registrar softmax_v2 caso o modelo use
def softmax_v2(x):
    return tf.nn.softmax(x)

tf.keras.utils.get_custom_objects()['softmax_v2'] = softmax_v2

# Carregar modelo MNIST treinado
modelo = load_model("modelo_mnist_novo.keras", compile=False)

app = Flask(__name__)

# Pasta de upload
offset = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(offset, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


def preprocessar_imagem(caminho):
    import cv2
    import numpy as np

    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

    # Remover ruído
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Normalizar contraste
    img = cv2.equalizeHist(img)

    # Redimensionar para 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Inverter se fundo for escuro (MNIST usa dígito claro em fundo escuro)
    if np.mean(img) > 127:
        img = 255 - img

    # Normalizar
    img = img.astype("float32") / 255.0

    # Formato correto para o modelo CNN (N, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    return img



@app.route('/upload', methods=['POST'])
def upload():
    if 'imagem' not in request.files:
        return "Nenhuma imagem enviada."

    file = request.files['imagem']

    if file.filename == '':
        return "Arquivo inválido."

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    try:
        entrada = preprocessar_imagem(img_path)

        # Predição
        pred = modelo.predict(entrada)
        numero = int(np.argmax(pred))

    except Exception as e:
        return f"Erro ao processar: {e}"

    return render_template("resultado.html", texto=str(numero))


@app.route('/resultado')
def resultado():
    texto = request.args.get('texto', '')
    return render_template('resultado.html', texto=texto)


if __name__ == '__main__':
    app.run(debug=True)
