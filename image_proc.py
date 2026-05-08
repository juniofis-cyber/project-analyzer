"""
Módulo de processamento de imagens para filmes radiocrômicos EBT3/EBT4.

Leitura de imagens TIFF/PNG/JPG preservando profundidade de bits (16-bit),
extração do canal vermelho e cálculo de densidade óptica / reflectância.
"""

import numpy as np
from PIL import Image
import cv2
import tifffile as tiff
from pathlib import Path


def read_image(path: str | Path, preserve_16bit: bool = True) -> np.ndarray:
    """
    Lê imagem de filme radiocrômico preservando a profundidade de bits original.

    Estratégia:
    1. Tenta tifffile primeiro (melhor para TIFF 16-bit)
    2. Fallback para OpenCV com flag -1 (IMREAD_UNCHANGED)
    3. Último fallback para PIL

    Returns:
        np.ndarray: Imagem em formato array NumPy. Para 16-bit, dtype será uint16.
    """
    path = Path(path)

    if path.suffix.lower() in ('.tif', '.tiff'):
        try:
            img = tiff.imread(str(path))
            if img is not None and img.ndim >= 2:
                return np.asarray(img)
        except Exception:
            pass

    # OpenCV: -1 carrega sem conversão, preservando 16-bit se presente
    img_cv = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img_cv is not None:
        return np.asarray(img_cv)

    # Fallback PIL
    with Image.open(path) as pil_img:
        return np.array(pil_img)


def get_red_channel(image: np.ndarray, bgr_format: bool = True) -> np.ndarray:
    """
    Extrai o canal vermelho da imagem.

    Para filmes EBT3/EBT4, o canal vermelho é o mais sensível à radiação.
    Se a imagem estiver em BGR (OpenCV default), o vermelho é o canal 2.
    Se estiver RGB, o vermelho é o canal 0.

    Args:
        image: Array NumPy (H, W) ou (H, W, C)
        bgr_format: True se a imagem veio do OpenCV (BGR), False se RGB

    Returns:
        np.ndarray: Canal vermelho como array 2D
    """
    if image.ndim == 2:
        # Já é grayscale — assumir que representa o canal de interesse
        return image.astype(np.float64)

    if image.ndim == 3:
        if bgr_format and image.shape[2] >= 3:
            return image[:, :, 2].astype(np.float64)
        elif not bgr_format and image.shape[2] >= 3:
            return image[:, :, 0].astype(np.float64)
        elif image.shape[2] == 1:
            return image[:, :, 0].astype(np.float64)

    raise ValueError(f"Formato de imagem não suportado: shape={image.shape}")


def normalize_to_16bit(image: np.ndarray) -> np.ndarray:
    """
    Normaliza a imagem para escala 16-bit (0-65535) se necessário.
    """
    if image.dtype == np.uint16:
        return image.astype(np.float64)
    if image.dtype == np.uint8:
        return image.astype(np.float64) * 257.0  # 65535/255 = 257

    # Para float ou outros, assumir que já está normalizada 0-1 ou 0-65535
    img_max = np.max(image)
    if img_max <= 1.0 + 1e-6:
        return image.astype(np.float64) * 65535.0
    return image.astype(np.float64)


def compute_reflectance(red_channel: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """
    Calcula a reflectância (R) a partir do canal vermelho.

    R = pixel_value / (2^bit_depth - 1)

    A reflectância varia de 0 a 1.
    """
    max_val = (2 ** bit_depth) - 1
    return red_channel / max_val


def compute_optical_density(red_channel: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """
    Calcula a Densidade Óptica (OD) a partir do canal vermelho.

    OD = -log10(R) = -log10(pixel / (2^bit_depth - 1))

    Returns:
        np.ndarray: Densidade óptica (valores maiores = maior escurecimento)
    """
    reflectance = compute_reflectance(red_channel, bit_depth)
    # Evitar log(0)
    reflectance = np.clip(reflectance, a_min=1e-6, a_max=1.0)
    return -np.log10(reflectance)


def compute_net_optical_density(
    irradiated_channel: np.ndarray,
    background_channel: np.ndarray,
    bit_depth: int = 16
) -> np.ndarray:
    """
    Calcula a Net Optical Density (NOD).

    NOD = OD_irradiado - OD_background
        = log10(R_background / R_irradiado)

    Args:
        irradiated_channel: Canal vermelho da imagem irradiada
        background_channel: Canal vermelho da imagem não-irradiada (controle)

    Returns:
        np.ndarray: Net Optical Density
    """
    r_irr = compute_reflectance(irradiated_channel, bit_depth)
    r_bg = compute_reflectance(background_channel, bit_depth)

    # Evitar divisão por zero ou log negativo
    r_irr = np.clip(r_irr, a_min=1e-6, a_max=1.0)
    r_bg = np.clip(r_bg, a_min=1e-6, a_max=1.0)

    nod = np.log10(r_bg / r_irr)
    return nod


def average_roi(image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    Calcula a média de uma região de interesse (ROI).
    """
    roi = image[y:y+h, x:x+w]
    return float(np.mean(roi))


def median_roi(image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    Calcula a mediana de uma região de interesse (ROI).
    """
    roi = image[y:y+h, x:x+w]
    return float(np.median(roi))
