"""
Script de teste para a API MNIST
Use este script para testar se a API está funcionando corretamente
"""
import requests
import json

# URL da API (ajuste conforme necessário)
API_URL = "http://localhost:8000"

def test_root():
    """Testa o endpoint raiz"""
    print("Testando endpoint raiz...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2)}")
    print()

def test_health():
    """Testa o endpoint de health check"""
    print("Testando health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict(image_path):
    """Testa o endpoint de predição com uma imagem"""
    print(f"Testando predição com imagem: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/png')}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Dígito previsto: {result['predicted_class']}")
            print(f"Confiança: {result['confidence']:.2%}")
            print(f"Probabilidades: {json.dumps(result['class_probabilities'], indent=2)}")
        else:
            print(f"Erro: {response.text}")
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {image_path}")
    except Exception as e:
        print(f"Erro ao fazer requisição: {e}")
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("Testes da API MNIST")
    print("=" * 50)
    print()
    
    # Testa endpoints básicos
    test_root()
    test_health()
    
    # Testa predição (descomente e ajuste o caminho da imagem)
    # test_predict("path/to/your/test_image.png")
    
    print("Para testar com uma imagem, execute:")
    print('  test_predict("caminho/para/sua/imagem.png")')

