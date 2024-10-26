import json
import rasterio
import numpy as np
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import List
from segmentation_mask_overlay import overlay_masks
from matplotlib.patches import Patch


class Fmask:
    def read_landsat_bands(self, tif_file: str) -> np.ndarray:
        """Read the tif file

        Args:
            tif_file (str): A string with the path to the tif

        Returns:
            _type_: ndarray with all tif bands
        """
        with rasterio.open(tif_file) as src:
            bands = src.read()

        with open(tif_file.replace(".tif", ".json"), "r") as file:
            metadata = json.load(file)

        # print("Bands: ", len(bands))
        return bands, metadata

    def calculate_ndvi(self, b4: np.ndarray, b3: np.ndarray) -> List[np.ndarray]:
        """Calculate the NDVI spectral indice by: nir-red/nir+red

        Args:
            b4 (np.ndarray): Nir band for landsat4-5 TM and landsat7 ETM
            b3 (np.ndarray): Red band for landsat4-5 TM and landsat7 ETM

        Returns:
            List[np.ndarray]: A single band with the np.ndarray
        """

        ndvi = (b4 - b3) / (b4 + b3)

        modified_ndvi = ndvi.copy()

        b3_0_to_255 = np.max((b3 - np.min(b3)) / (np.max(b3) - np.min(b3)) * 255)

        pixels_to_modify = (b3_0_to_255 == 255) & (b4 > b3)

        modified_ndvi[pixels_to_modify] = 0

        return ndvi, modified_ndvi

    def calculate_ndsi(self, b2: np.ndarray, b5: np.ndarray) -> List[np.ndarray]:
        """_summary_

        Args:
            b2 (np.ndarray): _description_
            b5 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        ndsi = (b2 - b5) / (b2 + b5)

        modified_ndsi = ndsi.copy()

        b2_0_to_255 = np.max((b2 - np.min(b2)) / (np.max(b2) - np.min(b2)) * 255)

        # Saturation test
        pixels_to_modify = (b2_0_to_255 == 255) & (b5 > b2)

        modified_ndsi[pixels_to_modify] = 0

        return ndsi, modified_ndsi

    def calculate_ndwi(self, b3: np.ndarray, b5: np.ndarray) -> List[np.ndarray]:
        """Calculate the NDVI spectral indice by: nir-red/nir+red

        Args:
            b4 (np.ndarray): Nir band for landsat4-5 TM and landsat7 ETM
            b3 (np.ndarray): Red band for landsat4-5 TM and landsat7 ETM

        Returns:
            List[np.ndarray]: A single band with the np.ndarray
        """

        ndwi = (b3 - b5) / (b3 + b5)

        return ndwi

    def calculate_brightness_temperature(self, band_thermal, k1, k2, to_celsius=False):
        radiance = band_thermal * 0.05518 + 1.2378  # Ajuste conforme necessário
        bt_kelvin = k2 / np.log((k1 / radiance) + 1)
        if to_celsius:
            bt_celsius = bt_kelvin - 273.15
            return bt_celsius
        else:
            return bt_kelvin

    def basic_test(self, b7, bt, ndvi, ndsi) -> np.ndarray:
        """_summary_

        Args:
            b7 (_type_): _description_
            bt (_type_): _description_
            ndvi (_type_): _description_
            ndsi (_type_): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 1
        # bt = calculate_brightness_temperature(b6, 607.76, 1260.56, to_celsius=True)
        result = np.logical_and(b7 > 0.03, bt < 27)
        result = np.logical_and(result, ndvi < 0.8)
        result = np.logical_and(result, ndsi < 0.8)

        return result

    def calculate_mean_visible(
        self, b3: np.ndarray, b2: np.ndarray, b1: np.ndarray
    ) -> np.ndarray:
        """_summary_

        Args:
            b3 (np.ndarray): _description_
            b2 (np.ndarray): _description_
            b1 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        return (b3 + b2 + b1) / 3

    def whiteness_test(
        self, b3: np.ndarray, b2: np.ndarray, b1: np.ndarray
    ) -> np.ndarray:
        """_summary_

        Args:
            b3 (np.ndarray): _description_
            b2 (np.ndarray): _description_
            b1 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 2
        bands = np.array([b3, b2, b1])
        mean_visible = self.calculate_mean_visible(b3, b2, b1)
        whiteness = np.zeros_like(b3)

        for band in bands:
            # whiteness += np.abs(np.divide((np.subtract(band, mean_visible)), mean_visible))
            whiteness += np.abs((band - mean_visible) / mean_visible)

        return whiteness, whiteness < 0.8

    def hot_test(self, b1: np.ndarray, b3: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            b1 (np.ndarray): _description_
            b3 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 3
        # return np.multiply(np.subtract(b1, 0.5), np.subtract(b3, 0.08)) > 0
        b3 = 0.5 * b3
        hot_test = b1 - b3 - 0.08
        hot_test = hot_test > 0
        return hot_test

    def b4_over_b5_test(self, b4: np.ndarray, b5: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            b4 (np.ndarray): _description_
            b5 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 4
        return np.divide(b4, b5) > 0.75

    def pass_one(
        self,
        b1: np.ndarray,
        b3: np.ndarray,
        b4: np.ndarray,
        b5: np.ndarray,
        b7: np.ndarray,
        bt: np.ndarray,
        ndvi: np.ndarray,
        ndsi: np.ndarray,
        whiteness_test: np.ndarray,
    ):
        # plt.figure()
        # plt.imshow(whiteness_test)
        # plt.show()
    
        # Eq. 6
        pcp = np.logical_and(self.basic_test(b7, bt, ndvi, ndsi), whiteness_test)
        # pcp = np.logical_and(pcp, self.hot_test(b1, b3))
        pcp = np.logical_and(pcp, self.b4_over_b5_test(b4, b5))

        return pcp

    def water_test(self, ndvi: np.ndarray, b4: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            ndvi (np.ndarray): _description_
            b4 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 5
        return np.logical_or(
            np.logical_and(ndvi < 0.01, b4 < 0.11),
            np.logical_and(ndvi < 0.1, b4 < 0.05),
        )

    def clear_sky_water_test(self, water_test: np.array, b7: np.ndarray) -> np.ndarray:
        return np.logical_and(water_test, b7 < 0.03)

    def water_cloud_prob(
        self, water_test: np.ndarray, b5: np.ndarray, b7: np.ndarray, bt: np.ndarray
    ) -> np.ndarray:
        """_summary_

        Args:
            water_test (np.ndarray): _description_
            b5 (np.ndarray): _description_
            b7 (np.ndarray): _description_
            bt (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 7
        clear_sky_water = self.clear_sky_water_test(water_test=water_test, b7=b7)

        # Eq. 8
        t_water = None
        try:
            t_water = np.percentile(bt[clear_sky_water], 82.5)
        except:
            t_water = np.zeros_like(bt)

        # Eq. 9
        w_temperature_prob = (t_water - bt) / 4

        # Eq. 10
        brightness_prob = np.minimum(b5, 0.11) / 0.11

        # Eq. 11
        w_cloud_prob = w_temperature_prob * brightness_prob

        return w_cloud_prob

    def land_cloud_prob(
        self,
        bt: np.ndarray,
        modified_ndvi: np.ndarray,
        modified_ndsi: np.ndarray,
        whiteness: np.ndarray,
        clear_sky_land: List[np.ndarray],
    ):
        """_summary_

        Args:
            bt (np.ndarray): _description_
            modified_ndvi (np.ndarray): _description_
            modified_ndsi (np.ndarray): _description_
            whiteness (np.ndarray): _description_
            clear_sky_land (np.ndarray): _description_

        Returns:
            _type_: _description_
        """

        # Eq. 13
        try:
            t_low = np.percentile(bt[clear_sky_land], 17.5)
        except:
            t_low = 20

        try:
            t_high = np.percentile(bt[clear_sky_land], 82.5)
        except:
            t_high = 60

        # Eq. 14
        l_temperature_prob = (t_high + 4 - bt) / (t_high + 4 - (t_low - 4))

        # Eq. 15
        maximum = np.maximum(np.abs(modified_ndvi), np.abs(modified_ndsi))
        variability_prob = 1 - np.maximum(maximum, whiteness)
        # variability_prob = 1 - maximum
        # Eq. 16
        l_cloud_prob = l_temperature_prob * variability_prob

        return l_cloud_prob, t_low, t_high

    def pass_two(
        self,
        b5: np.ndarray,
        b7: np.ndarray,
        bt: np.ndarray,
        pcp: np.ndarray,
        modified_ndvi: np.ndarray,
        modified_ndsi: np.ndarray,
        water_test: np.ndarray,
        whiteness: np.ndarray,
    ) -> np.ndarray:
        """_summary_

        Args:
            b5 (np.ndarray): _description_
            b7 (np.ndarray): _description_
            bt (np.ndarray): _description_
            pcp (np.ndarray): _description_
            water_test (np.ndarray): _description_
            whiteness (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        w_cloud_prob = self.water_cloud_prob(water_test=water_test, b5=b5, b7=b7, bt=bt)

        # Eq. 12
        clear_sky_land = np.logical_not(pcp) & np.logical_not(water_test)

        l_cloud_prob, t_low, t_high = self.land_cloud_prob(
            bt=bt,
            modified_ndvi=modified_ndvi,
            modified_ndsi=modified_ndsi,
            whiteness=whiteness,
            clear_sky_land=clear_sky_land,
        )
        # Eq. 17 Problem here
        try:
            land_threshold = np.percentile(l_cloud_prob[clear_sky_land], 82.5) + 0.2
        except:
            land_threshold = 0.5
        # land_threshold = np.percentile(l_cloud_prob[clear_sky_land], 10) #+ 0.2
        # land_threshold = np.percentile(l_cloud_prob[clear_sky_land], 92.5) + 0.2
        # land_threshold = np.percentile(l_cloud_prob[clear_sky_land], 42.5) + 0.2
        print("Land threshold: ", land_threshold)
        # Eq. 18
        # Clouds above water
        # pcl_1 = pcp & water_test & (w_cloud_prob > 0.5)
        pcl_1 = np.logical_and(pcp, water_test)
        pcl_1 = np.logical_and(pcl_1, w_cloud_prob > 0.5)

        # Clouds above land
        # pcl_2 = pcp & (water_test == False) & (l_cloud_prob > land_threshold)
        pcl_2 = np.logical_and(pcp, np.logical_not(water_test))
        pcl_2 = np.logical_and(pcl_2, w_cloud_prob > land_threshold)


        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(l_cloud_prob > land_threshold)
        # plt.subplot(1, 2, 2)
        # plt.imshow(w_cloud_prob, cmap="rainbow")
        # plt.show()

        # High land cloud probability
        pcl_3 = np.logical_and(l_cloud_prob > 0.99, np.logical_not(water_test))
        # pcl_3 = (l_cloud_prob > 0.99) & (water_test == False)

        # temperature test
        pcl_4 = bt < (t_low - 35)

        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(bt)
        # plt.show()

        # pcl =  np.logical_or(np.logical_or(np.logical_or(pcl_1, pcl_2), pcl_3), pcl_4)
        pcl = pcl_1 | pcl_2 | pcl_3 | pcl_4
        return pcl

    def detect_clouds(
        self,
        b1: np.ndarray,
        b3: np.ndarray,
        b4: np.ndarray,
        b5: np.ndarray,
        b7: np.ndarray,
        bt: np.ndarray,
        ndvi: np.ndarray,
        ndsi: np.ndarray,
        modified_ndvi: np.ndarray,
        modified_ndsi: np.ndarray,
        whiteness_test: np.ndarray,
        whiteness: np.ndarray,
        water: np.ndarray,
    ) -> np.ndarray:
        # Pass One to get potencial cloud pixels
        pcp = self.pass_one(
            b1=b1,
            b3=b3,
            b4=b4,
            b5=b5,
            b7=b7,
            bt=bt,
            ndvi=ndvi,
            ndsi=ndsi,
            whiteness_test=whiteness_test,
        )

        # Pass Two returns potencial cloud layer
        pcl = self.pass_two(
            b5=b5,
            b7=b7,
            bt=bt,
            pcp=pcp,
            modified_ndvi=modified_ndvi,
            modified_ndsi=modified_ndsi,
            water_test=water,
            whiteness=whiteness,
        )

        return pcl

    def flood_fill_transformation(self, band: np.ndarray):
        # Normalizar a banda para o intervalo [0, 255]
        band4_normalized = (
            (band - np.min(band)) / (np.max(band) - np.min(band)) * 255
        ).astype(np.uint8)
        normalized_image = Image.fromarray(band4_normalized)

        # Inverter a imagem para que o flood-fill funcione corretamente (transformação morfológica)
        inverted_image = Image.fromarray(255 - band4_normalized)

        # Definir um ponto de partida fora da área de interesse (por exemplo, canto da imagem)
        seed_point = (0, 0)

        # Aplicar o flood-fill usando ImageDraw.floodfill
        ImageDraw.floodfill(inverted_image, xy=seed_point, value=255, thresh=00)

        # Inverter a imagem de volta ao original
        filled_image = Image.fromarray(255 - np.array(inverted_image))

        # Converter o resultado de volta para um array NumPy
        result = np.maximum(band4_normalized, np.array(filled_image))

        # Converter o resultado de volta para uma imagem PIL
        result_image = Image.fromarray(result.astype(np.uint8)//255)

        return result_image

    def detect_shadows(self, b4: np.ndarray, water_test: np.ndarray):
        flood_fill_b4 = self.flood_fill_transformation(b4)        
        # PCSL(Potential Cloud Shadow Layer) test

        return (flood_fill_b4 - b4 > -0.22) & np.logical_not(water_test)
        # fig = plt.figure(figsize=(25, 15))
        # plt.imshow(((flood_fill_b4 - b4) < 50) & np.logical_not(water_test), cmap='gray')
        # plt.title("Flood fill")
        # plt.show()

        # return ((flood_fill_b4 - b4) < 25) & np.logical_not(water_test)

    def save_tif(self, band: np.ndarray, tif_file: str, output_file: str) -> None:
        """_summary_

        Args:
            band (np.ndarray): _description_
            tif_file (str): _description_
            output_file (str): _description_
        """

        with rasterio.open(tif_file) as src:
            profile = src.profile
            profile.update(count=1)
            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(band, 1)

    def save_plot(
        self, masks: list, color_composite: np.ndarray, save_dir: str, name: str
    ) -> None:
        """Save a plot contains mask with mask

        Args:
            mask (np.ndarray): Final mask
            color_composite (str): A color composite with three bands
            save_dir (str): Directory to save the plot
            name (str): Name of the file
        """

        fig = plt.figure(figsize=(25, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(color_composite, interpolation="nearest", aspect="auto")
        plt.axis(False)

        colors = ["gold", "red", "blue"]
        classes = ["Nuvem", "Sombra de Nuvem", "Água"]
        masked_image = overlay_masks(
            color_composite, np.stack(masks, -1), classes, colors=colors
        )

        plt.subplot(1, 2, 2)
        plt.imshow(color_composite, interpolation="nearest", aspect="auto")
        plt.imshow(masked_image, alpha=1, interpolation="nearest", aspect="auto")

        legend_elements = [
            Patch(facecolor=colors[i], edgecolor="black", label=f"{classes[i]}")
            for i in range(len(colors))
        ]

        plt.legend(handles=legend_elements, loc="upper right", fontsize=25)
        plt.axis(False)

        fig.savefig(save_dir + name, dpi=fig.dpi)

    def separar_componentes(self, mask):
        """
        Separa uma máscara binária em várias máscaras, cada uma contendo um componente conectado.
        Args:
        - mask (np.ndarray): Máscara binária de entrada.
        Returns:
        - list of np.ndarray: Lista de máscaras binárias, cada uma contendo um componente conectado.
        """
        labeled_mask, num_features = label(mask)
        masks_individuais = []

        for i in range(1, num_features + 1):
            componente_mask = (labeled_mask == i).astype(np.uint8)
            masks_individuais.append(componente_mask)

        return masks_individuais

    def calcular_direcao_sombra(self, sza, saa):
        """
        Calcula a direção da sombra com base nos ângulos solares.
        Args:
        - sza (np.ndarray): Mapa do ângulo zenital do sol em radianos.
        - saa (np.ndarray): Mapa do ângulo azimutal do sol em radianos.
        Returns:
        - np.ndarray: Matriz de direções da sombra com shape (altura, largura, 2).
        """
        direcao_x = -np.sin(saa)
        direcao_y = -np.cos(saa)
        return np.stack([direcao_x, direcao_y], axis=-1)

    def projetar_sombra(self, nuvem, direcao_sombra, distancia_sombra, shape):
        """
        Projeta a sombra da nuvem na imagem.
        Args:
        - nuvem (np.ndarray): Máscara binária da nuvem.
        - direcao_sombra (np.ndarray): Matriz de direções da sombra com shape (altura, largura, 2).
        - distancia_sombra (float): Distância da sombra em pixels.
        - shape (tuple): Forma da imagem (altura, largura).
        Returns:
        - np.ndarray: Máscara binária da sombra projetada.
        """
        projecao_sombra = np.zeros(shape, dtype=np.uint8)
        pixels_nuvem = np.argwhere(nuvem)

        for pixel in pixels_nuvem:
            y, x = pixel
            dy, dx = direcao_sombra[y, x] * distancia_sombra
            y_proj = int(round(y + dy))
            x_proj = int(round(x + dx))

            if 0 <= y_proj < shape[0] and 0 <= x_proj < shape[1]:
                projecao_sombra[y_proj, x_proj] = 1

        return projecao_sombra

    def calcular_similaridade(self, projecao_sombra, sombra_real):
        """
        Calcula a similaridade entre a projeção da sombra e a sombra real.
        Args:
        - projecao_sombra (np.ndarray): Máscara binária da sombra projetada.
        - sombra_real (np.ndarray): Máscara binária da sombra real.
        Returns:
        - float: Similaridade entre a projeção da sombra e a sombra real.
        """
        interseccao = np.sum(projecao_sombra & sombra_real)
        area_projecao = np.sum(projecao_sombra)
        if area_projecao == 0:
            return 0
        return interseccao / area_projecao

    def encontrar_correspondencia_intervalo_altura(self, clouds, cloud_shadows, saa_map, sza_map, h_cloud_base, h_cloud_top, shape, num_passos=10, limiar_similaridade=0.2):
        """
        Encontra a correspondência de sombras de nuvens para um intervalo de alturas.
        Args:
        - clouds (list of np.ndarray): Lista de máscaras binárias de nuvens.
        - cloud_shadows (list of np.ndarray): Lista de máscaras binárias de sombras de nuvens.
        - saa_map (np.ndarray): Mapa do ângulo azimutal do sol em radianos.
        - sza_map (np.ndarray): Mapa do ângulo zenital do sol em radianos.
        - h_cloud_base (float): Altura mínima da nuvem em metros.
        - h_cloud_top (float): Altura máxima da nuvem em metros.
        - shape (tuple): Forma das imagens de entrada (altura, largura).
        - num_passos (int): Número de passos para discretizar o intervalo de alturas.
        - limiar_similaridade (float): Limiar de similaridade para correspondência de sombra.
        Returns:
        - np.ndarray: Máscara binária das sombras correspondentes.
        """
        # Converter ângulos para radianos
        saa_map = np.radians(saa_map)
        sza_map = np.radians(sza_map)

        # Máscaras zeradas para sombras correspondentes
        sombras_correspondentes = np.zeros(shape, dtype=np.uint8)

        # Alturas para considerar no intervalo
        alturas = np.linspace(h_cloud_base, h_cloud_top, num=num_passos)

        direcao_sombra = self.calcular_direcao_sombra(sza_map, saa_map)

        for nuvem in clouds:
            melhor_similaridade = 0
            melhor_sombra = None

            for altura in alturas:
                print(altura)
                distancia_sombra = altura * np.tan(sza_map.mean())  # Aproximação usando o ângulo zenital do sol médio
                projecao_sombra = self.projetar_sombra(nuvem, direcao_sombra, distancia_sombra, shape)

                # plt.figure()
                # plt.imshow(projecao_sombra)
                # plt.show()
                if np.max(projecao_sombra) == 0:
                    continue


                for sombra in cloud_shadows:
                    similaridade = self.calcular_similaridade(projecao_sombra, sombra)
                    if similaridade == 0:
                        continue

                    if similaridade > melhor_similaridade:
                        melhor_similaridade = similaridade
                        melhor_sombra = sombra

                # print("best: ", melhor_similaridade, "limiar: ", limiar_similaridade)
                if melhor_similaridade >= limiar_similaridade:
                    # print('add')
                    sombras_correspondentes = np.logical_or(sombras_correspondentes, melhor_sombra).astype(np.uint8)
                    # plt.figure()
                    # plt.imshow(sombras_correspondentes)
                    # plt.show()
                # print("")
        return sombras_correspondentes

    def create_fmask(self, tif_file: str) -> np.ndarray:
        """Receives the landsat image and return the segmentation mask for
           cloud and cloud shadow

        Args:
            tif_file (str): Path to .tif with the bands

        Returns:
            np.ndarray: Mask containing cloud segmentation(value 1)
                        and cloud shadow (value 2)
        """
        # Open .tif
        bands, metadata = self.read_landsat_bands(tif_file)

        # Extract each band
        B1 = bands[0]
        B2 = bands[1]
        B3 = bands[2]
        B4 = bands[3]
        B5 = bands[4]
        B6 = bands[5]
        B7 = bands[6]

        saa_band = bands[9]
        sza_band = bands[10]
        vaa_band = bands[11]
        vza_band = bands[12]


        # Calculation the necessary indices
        ndvi, modified_ndvi = self.calculate_ndvi(B4, B3)
        ndsi, modified_ndsi = self.calculate_ndsi(B2, B5)
        ndwi = self.calculate_ndwi(B3, B5)
        bt = B6 - 273.15
        whiteness, whiteness_test = self.whiteness_test(B3, B2, B1)
        water_test = self.water_test(ndvi, B7)

        # Get cloud mask
        cloud_mask = self.detect_clouds(
            b1=B1,
            b3=B3,
            b4=B4,
            b5=B5,
            b7=B7,
            bt=bt,
            ndvi=ndvi,
            ndsi=ndsi,
            modified_ndvi=modified_ndvi,
            modified_ndsi=modified_ndsi,
            whiteness=whiteness,
            whiteness_test=whiteness_test,
            water=water_test,
        )

        # Get shadow cloud mask
        shadow_mask = self.detect_shadows(B4, water_test=water_test)

        # Exemplo de uso
        # Suponha que cloud_mask e shadow_mask sejam suas máscaras binárias de nuvens e sombras

        # Máscaras de nuvens e sombras segmentadas
        cloud_masks_individuais = self.separar_componentes(cloud_mask)
        shadow_masks_individuais = self.separar_componentes(shadow_mask)

        # Printando o número de nuvens e sombras encontradas
        print(f"Número de nuvens encontradas: {len(cloud_masks_individuais)}")
        print(f"Número de sombras encontradas: {len(shadow_masks_individuais)}")

        # Exemplo de uso
        # nuvens e sombras são listas de máscaras binárias (arrays numpy)
        # shape é o formato das máscaras originais

        altura_base = 200  # Altura mínima da nuvem em metros
        altura_top = 1200
        shape = B1.shape

        # Encontrar correspondências e gerar máscaras finais
        shadow_mask = self.encontrar_correspondencia_intervalo_altura(
            clouds=cloud_masks_individuais,
            cloud_shadows=shadow_masks_individuais,
            saa_map=saa_band,
            sza_map=sza_band,
            # sun_elevation=metadata['properties']['SUN_ELEVATION'],
            # sun_azimuth=metadata['properties']['SUN_AZIMUTH'],
            h_cloud_base=altura_base,
            h_cloud_top=altura_top,
            shape=shape,
        )

        water_mask = np.logical_and(ndwi, water_test)
        # return ndwi, cloud_mask, shadow_mask
        return (
            np.transpose(np.array([bands[4], bands[3], bands[2]]), [1, 2, 0]),
            cloud_mask,
            shadow_mask,
            ndwi,
        )


if __name__ == "__main__":
    import os

    # root = './test_images'
    root = "./Seixas/TOA/2004"

    inputs = [f"{root}/{img}" for img in os.listdir(root) if ".tif" in img]
    # inputs = ['./2004/seixas_20041124.tif']
    # inputs = ['./2004/seixas_20040905.tif']

    save_dir = "./results/"

    # inputs = ['./Seixas_big_picture/TOA/2004/seixas_20040414.tif']
    # inputs = ['./Seixas/TOA/2004/seixas_20040905.tif']
    # inputs = ['seixas_20041124.tif']
    # save_dir = "./"

    fmask = Fmask()

    for inp in inputs:
        file_name = f'{inp.split("/")[-1].split(".")[0]}_result.png'
        color_composite, cloud_mask, shadow_mask, water_mask = fmask.create_fmask(inp)
        # fmask.save_tif(result, inputs[0], './test.tif')
        fmask.save_plot(
            [cloud_mask, shadow_mask, water_mask],
            color_composite,
            save_dir,
            name=file_name,
        )
        # fmask.save(result, inp, file_name)
