import argparse
import subprocess
import sys
from pathlib import Path

def run_training():
    """Ejecuta el script de entrenamiento"""
    try:
        # Asegúrate de que el archivo se llame 'train.py' o 'cifar.py' según corresponda
        subprocess.run([sys.executable, str(Path("src") / "cifar.py")], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en entrenamiento: {e}")

def run_webcam():
    """Ejecuta el clasificador con la webcam"""
    try:
        subprocess.run([sys.executable, str(Path("src") / "webcam.py")], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en webcam: {e}")

def main():
    parser = argparse.ArgumentParser(description="Clasificador Marcador vs Lápiz")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "webcam"],
        help="Modo de ejecución: 'train' (entrenamiento) o 'webcam' (detección)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("🚀 Iniciando entrenamiento...")
        run_training()
    elif args.mode == "webcam":
        print("📷 Iniciando clasificación por webcam...")
        run_webcam()

if __name__ == "__main__":
    main()