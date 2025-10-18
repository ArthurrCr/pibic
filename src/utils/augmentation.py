import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np


class CloudIndicesTransform(ImageOnlyTransform):
    """
    Calcula índices espectrais para detecção de nuvens usando bandas Sentinel-2.
    
    Ordem das bandas:
    0: B01 (Coastal aerosol), 1: B02 (Blue), 2: B03 (Green), 3: B04 (Red),
    4: B05 (Red edge 1), 5: B06 (Red edge 2), 6: B07 (Red edge 3), 
    7: B08 (NIR), 8: B8A (Red edge 4), 9: B09 (Water vapor),
    10: B10 (Cirrus), 11: B11 (SWIR1), 12: B12 (SWIR2)
    """
    def __init__(
        self,
        compute_cdi: bool = True,
        compute_ci: bool = True,
        compute_bsi: bool = True,
        compute_hcci: bool = True,
        always_apply: bool = True,
        p: float = 1.0
    ):
        super().__init__(always_apply, p)
        self.compute_cdi = compute_cdi
        self.compute_ci = compute_ci
        self.compute_bsi = compute_bsi
        self.compute_hcci = compute_hcci
        
    def apply(self, image, **params):
        h, w, c = image.shape
        indices_list = []
        
        if c >= 13:
            # Extract bands and compute indices
            blue = image[:, :, 1].astype(np.float32)     # B02
            red = image[:, :, 3].astype(np.float32)      # B04
            nir = image[:, :, 7].astype(np.float32)      # B08
            cirrus = image[:, :, 10].astype(np.float32)  # B10
            swir1 = image[:, :, 11].astype(np.float32)   # B11
            swir2 = image[:, :, 12].astype(np.float32)   # B12
            
            # Compute indices (CI, CDI, etc.)
            if self.compute_cdi:
                cdi = (swir1 - red) / (swir1 + red + 1e-8)
                indices_list.append(cdi[:, :, np.newaxis])

            if self.compute_ci:
                ci = (nir - red) / (nir + red + 1e-8)
                indices_list.append(ci[:, :, np.newaxis])

            if self.compute_bsi:
                bsi = (blue - swir1) / (blue + swir1 + 1e-8)
                indices_list.append(bsi[:, :, np.newaxis])

            if self.compute_hcci:
                hcci = (swir2 - swir1) / (swir2 + swir1 + 1e-8)
                hcci_cirrus = cirrus / (cirrus + swir1 + 1e-8)
                hcci_combined = 0.7 * hcci + 0.3 * hcci_cirrus
                indices_list.append(hcci_combined[:, :, np.newaxis])
            
        if indices_list:
            indices = np.concatenate(indices_list, axis=2)
            return np.concatenate([image, indices], axis=2)
        
        return image

    
    def get_transform_init_args_names(self):
        return ("compute_cdi", "compute_ci", "compute_bsi", "compute_hcci")


def get_albumentations_transform(
    img_height: int = 512,
    img_width: int = 512,

    # Controle de redimensionamento
    apply_resize: bool = False,
    apply_random_resized_crop: bool = False,
    scale: tuple = (0.9, 1.0),

    # Transformações geométricas
    rotation_limit: float = 30.0,
    apply_horizontal_flip: bool = False,
    horizontal_flip_prob: float = 0.5,
    apply_vertical_flip: bool = False,
    vertical_flip_prob: float = 0.5,
    apply_shift_scale_rotate: bool = False,
    shift_limit: float = 0.05,
    scale_limit: float = 0.1,

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
    gaussian_blur_sigma: float = 1.0,

    # Sombras e realces aleatórios
    apply_random_shadows_highlights: bool = False,

    # GridDropout
    apply_grid_dropout: bool = False,
    grid_dropout_ratio: float = 0.2,

    # CoarseDropout
    apply_coarse_dropout: bool = False,
    coarse_dropout_max_holes: int = 5,
    coarse_dropout_max_size: int = 50,

    # ElasticTransform
    apply_elastic_transform: bool = False,
    elastic_alpha: float = 50,
    elastic_sigma: float = 5,

    # OpticalDistortion
    apply_optical_distortion: bool = False,
    distort_limit: float = 0.2,

    # CLAHE
    apply_clahe: bool = False,
    clahe_clip_limit: float = 2.0,

    # RandomGamma
    apply_random_gamma: bool = False,
    gamma_limit: tuple = (80, 120),

    # Simulações atmosféricas
    apply_random_fog: bool = False,
    apply_random_tone: bool = False,
    tone_scale: float = 0.3,

    # Índices espectrais para nuvens
    apply_cloud_indices: bool = True,
    compute_cdi: bool = True,
    compute_ci: bool = True,
    compute_bsi: bool = True,
    compute_hcci: bool = True,

    # Conversão para tensor
    apply_to_tensor: bool = True,
):
    transform_list = []

    # Redimensionamento ou recorte redimensionado
    if apply_resize:
        transform_list.append(A.Resize(height=img_height, width=img_width))

    if apply_random_resized_crop:
        transform_list.append(A.RandomResizedCrop(size=(img_height, img_width), scale=scale, ratio=(0.75, 1.33), p=1.0))

    # Transformações geométricas
    if rotation_limit > 0:
        transform_list.append(A.Rotate(limit=rotation_limit, p=0.7, border_mode=cv2.BORDER_CONSTANT))
    if apply_horizontal_flip:
        transform_list.append(A.HorizontalFlip(p=horizontal_flip_prob))
    if apply_vertical_flip:
        transform_list.append(A.VerticalFlip(p=vertical_flip_prob))

    if apply_shift_scale_rotate:
        transform_list.append(A.Affine(
            scale=(1-scale_limit, 1+scale_limit),
            translate_percent=(-shift_limit, shift_limit),
            rotate=(-rotation_limit, rotation_limit),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ))

    # Transformações fotométricas
    if apply_color_jitter:
        transform_list.append(A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=1.0))

    # Gaussian Blur
    if apply_gaussian_blur:
        transform_list.append(A.GaussianBlur(blur_limit=gaussian_blur_kernel, sigma_limit=(gaussian_blur_sigma, gaussian_blur_sigma), p=gaussian_blur_prob))

    # Sombras e realces aleatórios
    if apply_random_shadows_highlights:
        transform_list.append(A.RandomShadow(p=0.5))
        transform_list.append(A.RandomBrightnessContrast(p=0.5))

    # CLAHE
    if apply_clahe:
        transform_list.append(A.CLAHE(clip_limit=clahe_clip_limit, tile_grid_size=(8, 8), p=0.3))

    # RandomGamma
    if apply_random_gamma:
        transform_list.append(A.RandomGamma(gamma_limit=gamma_limit, p=0.3))

    # ElasticTransform
    if apply_elastic_transform:
        transform_list.append(A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            p=0.3
        ))

    # OpticalDistortion
    if apply_optical_distortion:
        transform_list.append(A.OpticalDistortion(
            distort_limit=distort_limit,
            p=0.3
        ))

    # GridDropout
    if apply_grid_dropout:
        transform_list.append(A.GridDropout(
            ratio=grid_dropout_ratio,
            random_offset=True,
            p=0.3
        ))

    # CoarseDropout
    if apply_coarse_dropout:
        transform_list.append(A.CoarseDropout(
            max_holes=coarse_dropout_max_holes,
            max_height=coarse_dropout_max_size,
            max_width=coarse_dropout_max_size,
            min_holes=1,
            min_height=coarse_dropout_max_size // 2,
            min_width=coarse_dropout_max_size // 2,
            fill_value=0,
            mask_fill_value=0,
            p=0.3
        ))

    # RandomFog
    if apply_random_fog:
        transform_list.append(A.RandomFog(p=0.2))

    if apply_random_tone:
        transform_list.append(A.RandomToneCurve(scale=tone_scale, p=0.2))

    # Índices espectrais - MOVIDO PARA LOGO ANTES DO ToTensorV2
    if apply_cloud_indices:
        transform_list.append(CloudIndicesTransform(
            compute_cdi=compute_cdi,
            compute_ci=compute_ci,
            compute_bsi=compute_bsi,
            compute_hcci=compute_hcci,
            p=1.0
        ))

    # Conversão para tensor - SEMPRE POR ÚLTIMO
    if apply_to_tensor:
        transform_list.append(ToTensorV2())

    return A.Compose(transform_list)