import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_albumentations_transform(
    img_height: int = 512,
    img_width: int = 512,
    mean=None,
    std=None,
    # Redimensionamento e crop
    apply_random_resized_crop: bool = False,
    scale: tuple = (0.8, 1.0),
    
    # Transformações geométricas
    rotation_limit: float = 30.0,
    apply_horizontal_flip: bool = False,
    horizontal_flip_prob: float = 0.5,
    
    # Transformações fotométricas
    apply_color_jitter: bool = False,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    
    # Gaussian Blur
    apply_gaussian_blur: bool = False,
    gaussian_blur_kernel: int = 3,
    gaussian_blur_prob: float = 0.5,
    gaussian_blur_sigma: float = 1.0
):
    """
    Retorna um pipeline de transformações para augmentação.
    Se apply_random_resized_crop=True, utiliza RandomResizedCrop (com o parâmetro "size").
    Caso contrário, realiza um Resize simples.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    transform_list = []
    
    # Redimensionamento ou RandomResizedCrop
    if apply_random_resized_crop:
        transform_list.append(
            A.RandomResizedCrop(
                size=(img_height, img_width),
                scale=scale,
                ratio=(0.75, 1.33),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0
            )
        )
    else:
        transform_list.append(
            A.Resize(height=img_height, width=img_width)
        )
    
    # Transformações geométricas
    if rotation_limit > 0:
        transform_list.append(
            A.Rotate(limit=rotation_limit, p=0.7, border_mode=cv2.BORDER_CONSTANT)
        )
    
    if apply_horizontal_flip:
        transform_list.append(
            A.HorizontalFlip(p=horizontal_flip_prob)
        )
    
    # Transformações fotométricas aplicadas apenas nas bandas RGB (supondo que elas estejam
    # nos índices 1, 2 e 3) se a imagem tiver 13 canais.
    if apply_color_jitter and any([brightness, contrast, saturation, hue]):
        def apply_rgb_color_jitter(image, **kwargs):
            # Se a imagem tem 13 canais, aplica o ColorJitter somente nas bandas RGB
            if image.shape[-1] == 13:
                # Aplica com probabilidade 70%
                if np.random.rand() < 0.7:
                    img_copy = image.copy()
                    # Extrai as bandas RGB (supondo que as bandas B2, B3 e B4 estejam nos índices 1, 2 e 3)
                    rgb = img_copy[:, :, [1, 2, 3]]
                    # Se o dtype não for uint8, converte (assumindo que seja uint16) para escala [0, 255]
                    if rgb.dtype != np.uint8:
                        rgb = ((rgb.astype(np.float32) / 65535.0) * 255).astype(np.uint8)
                    # Aplica o ColorJitter nos RGB
                    jitter = A.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue,
                        p=1.0
                    )
                    rgb_aug = jitter(image=rgb)['image']
                    # Converte de volta para o dtype original (revertendo a escala)
                    if image.dtype != np.uint8:
                        rgb_aug = ((rgb_aug.astype(np.float32) / 255.0) * 65535).astype(image.dtype)
                    # Substitui as bandas alteradas
                    img_copy[:, :, [1, 2, 3]] = rgb_aug
                    return img_copy
                else:
                    return image
            else:
                # Se a imagem não for 13 canais, aplica normalmente
                jitter = A.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                    p=0.7
                )
                return jitter(image=image)['image']
        
        transform_list.append(
            A.Lambda(image=apply_rgb_color_jitter, mask=lambda mask, **kwargs: mask)
        )
    
    # Gaussian Blur
    if apply_gaussian_blur:
        transform_list.append(
            A.GaussianBlur(
                blur_limit=gaussian_blur_kernel,
                sigma_limit=(gaussian_blur_sigma, gaussian_blur_sigma),
                p=gaussian_blur_prob
            )
        )
    
    # Normalização e conversão para tensor
    transform_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return A.Compose(transform_list)
