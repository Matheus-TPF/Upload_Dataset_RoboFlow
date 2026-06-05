import os
import requests
import time

from pathlib import Path
from contextlib import ExitStack
from dotenv import load_dotenv 

load_dotenv(".env")

# Troque os caminhos conforme for necessário
CONFIG = {
    "API_KEY": os.getenv("API_KEY"),
    "PATHS": {
        "train": Path(r"C:\Users\matheus.jose\Documents\TPF\Upload_Roboflow\ModelV5.3\ModelV5.3\train"),
        "valid": Path(r"C:\Users\matheus.jose\Documents\TPF\Upload_Roboflow\ModelV5.3\ModelV5.3\valid"),
        "test":  Path(r"C:\Users\matheus.jose\Documents\TPF\Upload_Roboflow\ModelV5.3\ModelV5.3\test"),
    },
    "WORKSPACE": os.getenv("WORKSPACE"),
    "LOG_FILE": "upload_history.log"
}

auth_check = requests.get(f"https://api.roboflow.com/{CONFIG['WORKSPACE']}?api_key={CONFIG['API_KEY']}")
print(f"Status de Permissão do Workspace: {auth_check.status_code}")
if auth_check.status_code in [401, 404, 403]:
    print("Verifique sua chave de API")

ENVIADOS = set()
if os.path.exists(CONFIG["LOG_FILE"]):
    with open(CONFIG["LOG_FILE"], "r") as f:
        ENVIADOS = set(f.read().splitlines())

def ja_enviado(nome_arquivo):
    return nome_arquivo in ENVIADOS

def registrar_sucesso(nome_arquivo):
    with open(CONFIG["LOG_FILE"], "a") as f:
        f.write(f"{nome_arquivo}\n")
    ENVIADOS.add(nome_arquivo) # Atualiza o set na memória

def upload_imagem_roboflow(img_path, label_path, split):
    url = "https://api.roboflow.com/upload"
    params = {
        "api_key": CONFIG["API_KEY"],
        "name": img_path.name,
        "split": split,
        "annotation_type": "yolo",
        "annotation_filename": label_path.name if label_path else None
    }
    
    try:
        with ExitStack() as stack:
            files = {"file": stack.enter_context(open(img_path, "rb"))}
            if label_path and label_path.exists():
                files["annotation"] = stack.enter_context(open(label_path, "rb"))
            
            response = requests.post(url, params=params, files=files, timeout=30)
            return response
    except Exception as e:
        print(f"Erro crítico ao processar {img_path.name}: {e}")
        return None

def processar_lotes(split_name, tamanho_lote=20):
    caminho = CONFIG["PATHS"].get(split_name)
    imagens = list((caminho / "images").glob("*.jpg"))
    
    print(f"Total de imagens encontradas: {len(imagens)}")

    for i in range(0, len(imagens), tamanho_lote):
        lote = imagens[i:i + tamanho_lote]
        print(f"\n>>> Iniciando Lote {i//tamanho_lote + 1} | Imagens {i} a {i+len(lote)}")

        for img in lote:
            if ja_enviado(img.name): continue
            
            label = caminho / "labels" / (img.stem + ".txt")
            resp = upload_imagem_roboflow(img, label if label.exists() else None, split_name)
            
            if resp and resp.status_code == 200:
                registrar_sucesso(img.name)
                print(f"[OK] {img.name}")
            else:
                print(f"[ERRO] {img.name} - Status: {resp.status_code if resp else 'N/A'}")

        print(f"--- Lote {i//tamanho_lote + 1} finalizado. Aguardando 2s para próxima rodada ---")
        time.sleep(2) 



if __name__ == "__main__":

    if not CONFIG["API_KEY"]:
        print("ERRO: API_KEY não configurada no arquivo .env")
    else:
        processar_lotes("train")
        # processar_split("valid")
        # processar_split("test")

