import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_albumentations_transform(
    img_height: int = 384,
    img_width: int = 384,
    mean=None,
    std=None,
    # Parâmetros de rotação
    rotation_limit: float = 180.0,  # Limite de rotação em graus (-rotation_limit, +rotation_limit)
    
    # Parâmetros de color jitter
    apply_color_jitter: bool = False,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    
    # Redimensionamento e crop
    apply_random_resized_crop: bool = False,
    scale: tuple = (0.8, 1.0),
    
    # Flip horizontal
    apply_horizontal_flip: bool = False,
    horizontal_flip_prob: float = 0.5,
    
    # Zoom
    apply_zoom: bool = False,
    zoom_factor: float = 1.0,  # 1.0 = sem zoom
    zoom_prob: float = 0.5,    # Probabilidade de aplicar o zoom
    
    # Gaussian Blur
    apply_gaussian_blur: bool = False,
    gaussian_blur_kernel: int = 3,
    gaussian_blur_prob: float = 0.5,
    gaussian_blur_sigma: float = 1.0
):
    """
    Retorna um pipeline de transformações com base nos parâmetros fornecidos.
    Todas as transformações geométricas são aplicadas simultaneamente na imagem e na máscara.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    transform_list = []

    # Redimensionamento ou RandomResizedCrop
    if apply_random_resized_crop:
        transform_list.append(A.RandomResizedCrop(height=img_height, width=img_width, scale=scale))
    else:
        transform_list.append(A.Resize(height=img_height, width=img_width))
    
    # Rotação Arbitrária
    if rotation_limit > 0:
        transform_list.append(A.Rotate(limit=rotation_limit, p=1.0, border_mode=0))  # border_mode=0 -> preenchimento com preto
    
    # Color Jitter (Brightness, Contrast, Saturation, Hue)
    if apply_color_jitter and (brightness > 0 or contrast > 0 or saturation > 0 or hue > 0):
        transform_list.append(A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1.0
        ))
    
    # Zoom
    if apply_zoom and zoom_factor != 1.0:
        # A função RandomScale altera o tamanho da imagem, mas para um zoom fixo,
        # podemos usar A.Affine com scale
        transform_list.append(A.RandomScale(scale=(zoom_factor, zoom_factor), p=zoom_prob))
    
    # Flip Horizontal
    if apply_horizontal_flip and horizontal_flip_prob > 0:
        transform_list.append(A.HorizontalFlip(p=horizontal_flip_prob))
    
    # Gaussian Blur
    if apply_gaussian_blur and gaussian_blur_prob > 0:
        transform_list.append(
            A.GaussianBlur(blur_limit=gaussian_blur_kernel, sigma_limit=gaussian_blur_sigma, p=gaussian_blur_prob)
        )

    
    # Normalização e conversão para tensor
    transform_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    augmentation_transform = A.Compose(transform_list)
    return augmentation_transform