import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from skimage.morphology import remove_small_objects
from typing import Tuple

class Fmask:
    """
    Implementação do algoritmo Fmask adaptado para Sentinel-2 com melhorias:
    - Correção de índices espectrais (NDVI, NDSI, NDWI) para separar vegetação, neve e água.
    - Detecção aprimorada de água e sombras (NDWI, projeção geométrica, validação espectral).
    - Consideração de ângulos solares (zenith e azimuth) e altura média das nuvens para projetar sombras.
    - Operações morfológicas (remoção de pequenos objetos e dilatação) no pós-processamento.
    - Ajustes de limiares para reduzir falsos positivos em superfícies claras (areia, neve, construções) e escuras (água, vegetação densa).
    - Possibilidade de estender para usar banda de Cirrus (B10) ou índices complementares (CDI, HOT) em cenários mais complexos.
    """

    def calculate_ndvi(self, nir: np.ndarray, red: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula NDVI e versão modificada para pixels saturados.
        
        NDVI = (NIR - Red) / (NIR + Red)
        Usado para distinguir vegetação (valores altos) de outras superfícies (baixas).
        """
        ndvi = self._safe_divide(nir - red, nir + red)

        # Detecção de saturação no Vermelho
        red_normalized = self._normalize_band(red)
        saturated_red = (red_normalized == 1.0) & (nir > red)

        modified_ndvi = ndvi.copy()
        modified_ndvi[saturated_red] = -1  # Marca como inválido ou suspeito

        return ndvi, modified_ndvi

    def calculate_ndsi(self, green: np.ndarray, swir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula NDSI com tratamento de pixels saturados.
        
        NDSI = (Green - SWIR) / (Green + SWIR)
        Ajuda a separar neve (NDSI alto) de nuvens (NDSI médio/baixo).
        """
        ndsi = self._safe_divide(green - swir, green + swir)

        # Detecção de saturação no Verde
        green_normalized = self._normalize_band(green)
        saturated_green = (green_normalized == 1.0) & (swir > green)

        modified_ndsi = ndsi.copy()
        modified_ndsi[saturated_green] = -1  # Marca como inválido ou suspeito

        return ndsi, modified_ndsi

    def calculate_ndwi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        NDWI para detecção de água.
        
        NDWI = (Green - NIR) / (Green + NIR)
        """
        return self._safe_divide(green - nir, green + nir)

    def _basic_cloud_test(
        self,
        swir2: np.ndarray,
        ndvi: np.ndarray,
        ndsi: np.ndarray,
        swir2_thresh: float,
        ndvi_thresh: float,
        ndsi_thresh: float
    ) -> np.ndarray:
        """
        Teste básico para pixels candidatos a nuvem com limiares ajustados.
        Exemplo de defaults:
         - swir2 > 0.07
         - ndvi < 0.5
         - ndsi < 0.5

        A lógica: nuvens refletem bem em SWIR2,
        e costumam ter NDVI e NDSI mais baixos do que neve/vegetação.
        """
        return (
            (swir2 > swir2_thresh) &
            (ndvi < ndvi_thresh) &
            (ndsi < ndsi_thresh)
        )

    def _calculate_whiteness(self, blue: np.ndarray, green: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        Calcula índice de 'brancura' para detecção de nuvens em bandas visíveis.
        Verifica quão similares são as bandas B2, B3, B4: nuvens tendem a refletir
        quase igualmente em todo espectro VIS, resultando em valor baixo de whiteness.
        """
        mean_visible = (blue + green + red) / 3.0
        eps = 1e-6  # Evita divisão por zero
        return (np.abs(blue - mean_visible) +
                np.abs(green - mean_visible) +
                np.abs(red - mean_visible)) / (mean_visible + eps)

    def _water_detection(self, ndwi: np.ndarray, swir1: np.ndarray) -> np.ndarray:
        """
        Detecção de água combinando NDWI e SWIR1.
        Ajuste os limites dependendo da cena, pois água em SWIR1
        normalmente é mais escura.
        """
        return (ndwi > 0.2) & (swir1 < 0.15)

    def _shadow_projection(self, cloud_mask: np.ndarray, zenith: float, azimuth: float, cloud_height: float) -> np.ndarray:
        """
        Projeção de sombras baseada em nuvens, ângulos solares e altura média da nuvem.
        Calcula o deslocamento da sombra em pixels, levando em conta a resolução (~10m).
        
        Dica: em aplicações mais avançadas, pode-se iterar em várias alturas de nuvem
        para encontrar a melhor correspondência entre manchas escuras e projeção,
        refinando a detecção de sombras.
        """
        # Se o ângulo zenital for inválido ou muito alto, não projetamos
        if zenith <= 0 or zenith >= 90:
            return np.zeros_like(cloud_mask, dtype=bool)

        # Comprimento da sombra (em metros)
        shadow_length = cloud_height * np.tan(np.deg2rad(zenith))

        # Converte para deslocamentos em x e y (em metros) considerando ângulo azimutal
        # Note que a convenção do azimute pode variar; ajuste se necessário
        dx_m = -shadow_length * np.sin(np.deg2rad(azimuth))
        dy_m = -shadow_length * np.cos(np.deg2rad(azimuth))

        # Converte deslocamentos para pixels (assumindo ~10m de resolução)
        px_x = int(round(dx_m / 10.0))
        px_y = int(round(dy_m / 10.0))

        shifted_mask = np.zeros_like(cloud_mask, dtype=bool)

        # Deslocamento em x (colunas)
        if px_x > 0:
            shifted_mask[:, px_x:] = cloud_mask[:, :-px_x]
        elif px_x < 0:
            shifted_mask[:, :px_x] = cloud_mask[:, -px_x:]
        else:
            shifted_mask |= cloud_mask

        # Deslocamento em y (linhas)
        if px_y > 0:
            shifted_mask[px_y:, :] |= cloud_mask[:-px_y, :]
        elif px_y < 0:
            shifted_mask[:px_y, :] |= cloud_mask[-px_y:, :]
        else:
            shifted_mask |= cloud_mask

        return shifted_mask

    def _postprocess_mask(self, mask: np.ndarray, min_size: int = 50, dilate_iterations: int = 1) -> np.ndarray:
        """
        Aplica operações morfológicas para limpeza da máscara:
         - Remove pequenos objetos (min_size)
         - Dilatação binária (iterations)
        """
        # Remove componentes conexas muito pequenas
        cleaned = remove_small_objects(mask, min_size=min_size)

        # Estrutura para dilatação (conexão 8)
        structure = generate_binary_structure(2, 2)

        # A dilatação suaviza bordas e conecta pequenas falhas
        if dilate_iterations > 0:
            cleaned = binary_dilation(cleaned, structure=structure, iterations=dilate_iterations)

        return cleaned

    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Divisão segura com tratamento de divisão por zero."""
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=np.abs(denominator) > 1e-6
        )

    def _normalize_band(self, band: np.ndarray) -> np.ndarray:
        """Normalização de banda para 0-1, ignorando NaNs."""
        return (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band) + 1e-6)

    def generate_cloud_mask(
        self,
        bands: dict,
        zenith: float,
        azimuth: float,
        cloud_height: float = 2000.0,
        shadow_threshold: float = 0.15,
        whiteness_threshold: float = 0.4,
        brightness_threshold: float = 0.2,
        swir2_thresh: float = 0.07,
        ndvi_thresh: float = 0.5,
        ndsi_thresh: float = 0.5,
        min_size: int = 50,
        dilate_iterations: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gera máscaras de nuvem, água e sombra para uma imagem Sentinel-2,
        com ajustes para reduzir falsos positivos sobre superfícies claras/escuras.

        Parâmetros:
            bands (dict): Dicionário com bandas necessárias (B2, B3, B4, B8, B11, B12).
            zenith (float): Ângulo zenital solar (graus).
            azimuth (float): Ângulo azimutal solar (graus).
            cloud_height (float): Altura média da nuvem em metros (p/ projeção de sombras).
            shadow_threshold (float): Limiar no NIR (B8) para marcar sombra (< este valor).
            whiteness_threshold (float): Limiar do índice de 'brancura'.
            brightness_threshold (float): Limiar mínimo de brilho médio (VIS) para ser nuvem.
            swir2_thresh (float): Limiar em SWIR2 (B12) no teste básico de nuvem.
            ndvi_thresh (float): Limiar no NDVI para o teste básico de nuvem.
            ndsi_thresh (float): Limiar no NDSI para o teste básico de nuvem.
            min_size (int): Tamanho mínimo de objeto (em pixels) para não ser removido.
            dilate_iterations (int): Iterações de dilatação binária no pós-processamento.

        Retorna:
            (cloud_mask, water_mask, shadow_mask) em formato booleano.
        """
        # Cálculo dos índices espectrais
        ndvi, mod_ndvi = self.calculate_ndvi(bands['B8'], bands['B4'])
        ndsi, mod_ndsi = self.calculate_ndsi(bands['B3'], bands['B11'])
        ndwi = self.calculate_ndwi(bands['B3'], bands['B8'])

        # Detecção de água (usando NDWI + SWIR1)
        water_mask = self._water_detection(ndwi, bands['B11'])

        # Cálculo de 'brancura' em bandas visíveis (B2, B3, B4)
        whiteness = self._calculate_whiteness(bands['B2'], bands['B3'], bands['B4'])

        # Teste básico de nuvem (SWIR2, NDVI, NDSI)
        basic_clouds = self._basic_cloud_test(
            swir2=bands['B12'],
            ndvi=mod_ndvi,
            ndsi=mod_ndsi,
            swir2_thresh=swir2_thresh,
            ndvi_thresh=ndvi_thresh,
            ndsi_thresh=ndsi_thresh
        )

        # Verifica se pixel é suficientemente brilhante no VIS
        mean_vis = (bands['B2'] + bands['B3'] + bands['B4']) / 3.0
        bright_enough = mean_vis > brightness_threshold

        # Combinação: pixel é nuvem se passa no teste básico, é "branco" (whiteness baixo),
        # é brilhante e não está em área de água
        cloud_candidates = (
            basic_clouds &
            (whiteness < whiteness_threshold) &
            bright_enough &
            ~water_mask
        )

        # Projeção de sombras a partir da máscara de nuvens
        projected_shadows = self._shadow_projection(
            cloud_mask=cloud_candidates,
            zenith=zenith,
            azimuth=azimuth,
            cloud_height=cloud_height
        )

        # Determina áreas potencialmente escuras (p.ex., < 0.15 em B8 e < 0.2 em B11)
        # que não sejam água e nem nuvem
        shadow_areas = (
            (bands['B8'] < shadow_threshold) &
            (bands['B11'] < 0.2) &
            ~water_mask &
            ~cloud_candidates
        )

        # Interseção entre projeção de sombra e áreas escuras
        shadow_mask = shadow_areas & projected_shadows

        # Pós-processamento morfológico (remoção de ruídos e dilatação)
        final_clouds = self._postprocess_mask(
            cloud_candidates, min_size=min_size, dilate_iterations=dilate_iterations
        )
        final_shadows = self._postprocess_mask(
            shadow_mask, min_size=min_size, dilate_iterations=dilate_iterations
        )

        return final_clouds, water_mask, final_shadows


def generate_mask_fmask(
    bands_13: np.ndarray,  # shape (H, W, 13) em reflectâncias [0..1]
    angle_zenith: float,
    angle_azimuth: float,
    cloud_height: float = 2000.0,
    shadow_threshold: float = 0.15,
    whiteness_threshold: float = 0.4,
    brightness_threshold: float = 0.2,
    swir2_thresh: float = 0.07,
    ndvi_thresh: float = 0.5,
    ndsi_thresh: float = 0.5,
    min_size: int = 50,
    dilate_iterations: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera máscara de nuvens e sombras para Sentinel-2 usando Fmask adaptado.
    Recebe o array de 13 bandas (B1..B12) em reflectâncias (0..1),
    bem como parâmetros de ângulo solar e limiares.

    Retorna:
        (cloud_mask, shadow_mask) como arrays binários (0 ou 1).
    """

    # Extração das principais bandas de interesse (resolução 10m ou 20m) a partir de bands_13
    # bands_13: (H, W, 13) nas posições:
    #  B01=0, B02=1, B03=2, B04=3, B05=4, B06=5, B07=6, B08=7, B8A=8, B09=9, B10=10, B11=11, B12=12
    b2_blue   = bands_13[:, :, 1]
    b3_green  = bands_13[:, :, 2]
    b4_red    = bands_13[:, :, 3]
    b8_nir    = bands_13[:, :, 7]
    b11_swir1 = bands_13[:, :, 11]
    b12_swir2 = bands_13[:, :, 12]

    # Dicionário para passar ao Fmask
    bands_dict = {
        'B2':  b2_blue,
        'B3':  b3_green,
        'B4':  b4_red,
        'B8':  b8_nir,
        'B11': b11_swir1,
        'B12': b12_swir2
    }

    fmask = Fmask()

    # Gera as máscaras de nuvem, água e sombra
    cloud_mask, water_mask, shadow_mask = fmask.generate_cloud_mask(
        bands=bands_dict,
        zenith=angle_zenith,
        azimuth=angle_azimuth,
        cloud_height=cloud_height,
        shadow_threshold=shadow_threshold,
        whiteness_threshold=whiteness_threshold,
        brightness_threshold=brightness_threshold,
        swir2_thresh=swir2_thresh,
        ndvi_thresh=ndvi_thresh,
        ndsi_thresh=ndsi_thresh,
        min_size=min_size,
        dilate_iterations=dilate_iterations
    )

    # Converte para 0/1 (uint8)
    cloud_mask_bin = cloud_mask.astype(np.uint8)
    shadow_mask_bin = shadow_mask.astype(np.uint8)

    return cloud_mask_bin, shadow_mask_bin
