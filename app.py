from flask import Flask, render_template, request
import pytesseract
from PIL import Image
from flask import redirect, url_for
import os

app = Flask(__name__)

# Pasta para uploads
offset = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(offset, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'imagem' not in request.files:
        return "Nenhuma imagem enviada"

    file = request.files['imagem']

    if file.filename == '':
        return "Arquivo inválido"

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Processar OCR
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        os.environ['TESSDATA_PREFIX'] = r"C:\\Program Files\\Tesseract-OCR\\tessdata"
    
        img = Image.open(img_path)
        texto = pytesseract.image_to_string(img, lang='por')
    except Exception as e:
        return f"Erro ao processar: {e}"

    return f"<h2>Texto reconhecido:</h2><pre>{texto}</pre>"

@app.route('/resultado')
def resultado():
    texto = request.args.get('texto', '')
    return render_template('resultado.html', texto=texto)

if __name__ == '__main__':
    app.run(debug=True)
