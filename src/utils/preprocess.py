import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

def combine_bands(red_path, green_path, blue_path):
    # Carregar as imagens das bandas
    red = Image.open(red_path)
    green = Image.open(green_path)
    blue = Image.open(blue_path)
    
    # Converter as imagens para arrays numpy
    red_array = np.array(red).astype(np.float32)
    green_array = np.array(green).astype(np.float32)
    blue_array = np.array(blue).astype(np.float32)
    
    # Definir um epsilon para evitar divisão por zero
    epsilon = 1e-8  # ou outro valor pequeno
    
    # Função para normalizar cada banda
    def normalize_band(band_array):
        band_min = band_array.min()
        band_max = band_array.max()
        denominator = band_max - band_min
        if denominator == 0:
            # Se todos os valores são iguais, definir a banda como zeros ou um valor constante
            normalized_array = np.zeros_like(band_array)
        else:
            normalized_array = (band_array - band_min) / denominator * 255.0
        return normalized_array

    # Normalizar cada banda individualmente
    red_array = normalize_band(red_array)
    green_array = normalize_band(green_array)
    blue_array = normalize_band(blue_array)
    
    # Converter arrays para uint8
    red_image = Image.fromarray(red_array.astype(np.uint8))
    green_image = Image.fromarray(green_array.astype(np.uint8))
    blue_image = Image.fromarray(blue_array.astype(np.uint8))
    
    # Mesclar as bandas em uma imagem RGB
    rgb_image = Image.merge('RGB', (red_image, green_image, blue_image))
    return rgb_image


def equalize_histogram(image):
    """
    Aplica equalização de histograma a uma imagem RGB.
    """
    r, g, b = image.split()
    r_eq = ImageOps.equalize(r)
    g_eq = ImageOps.equalize(g)
    b_eq = ImageOps.equalize(b)
    image_eq = Image.merge('RGB', (r_eq, g_eq, b_eq))
    return image_eq

def adjust_brightness_contrast(image, brightness_factor=1.0, contrast_factor=1.0):
    """
    Ajusta o brilho e o contraste da imagem.
    """
    enhancer_brightness = ImageEnhance.Brightness(image)
    image_enhanced = enhancer_brightness.enhance(brightness_factor)
    
    enhancer_contrast = ImageEnhance.Contrast(image_enhanced)
    image_enhanced = enhancer_contrast.enhance(contrast_factor)
    
    return image_enhanced
