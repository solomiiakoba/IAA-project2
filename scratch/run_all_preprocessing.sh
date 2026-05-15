#!/bin/bash

# Script para correr todas as configurações de pré-processamento com HOG ajustado
VENV_PYTHON="./venv/bin/python3"
SCRIPT="./src/preprocessing_v2.py"

# Configurações para testar: "Largura Altura HOG_PPC"
CONFIGS=(
    "1024 768 64"   # Alta Res, HOG Grosso
    "1024 768 128"  # Alta Res, HOG muito Grosso
    "512 384 32"    # Média Res, HOG Proporcional
    "512 384 64"    # Média Res, HOG Grosso
    "256 192 16"    # Baixa Res, HOG Proporcional
    "256 192 32"    # Baixa Res, HOG Grosso
    "128 128 16"    # Baseline Quadrado (Original)
    "128 128 8"     # Baseline Quadrado, HOG Fino
    "128 96 16"     # Retângulo Pequeno
    "64 48 8"       # Muito Pequeno
)

PARTS=("both")

for cfg in "${CONFIGS[@]}"; do
    set -- $cfg
    W=$1
    H=$2
    HOG=$3
    
    for part in "${PARTS[@]}"; do
        # 1. Sem Padding
        echo "------------------------------------------------"
        echo "A processar: ${W}x${H} | HOG PPC: ${HOG} | Método: Resize"
        $VENV_PYTHON $SCRIPT --width $W --height $H --hog_ppc $HOG --part $part
        
        # 2. Com Padding
        echo "------------------------------------------------"
        echo "A processar: ${W}x${H} | HOG PPC: ${HOG} | Método: Padding"
        $VENV_PYTHON $SCRIPT --width $W --height $H --hog_ppc $HOG --part $part --padding
    done
done

echo "Todos os processamentos concluídos!"
