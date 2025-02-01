import os
import numpy as np
from tqdm import tqdm
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import cv2  # Para utilizar CLAHE (Equalização de histograma adaptativa)
import random
from IPython.display import clear_output
from typing import List, Tuple


def create_file_pairs(processed_red_dir: str,
                      processed_green_dir: str,
                      processed_blue_dir: str,
                      processed_mask_dir: str) -> List[Tuple[str, str, str, str, str]]:
    """
    Retorna uma lista de tuplas com (base_name, arquivo_red, arquivo_green, arquivo_blue, arquivo_mask)
    correspondentes às imagens que existem em todos os quatro diretórios.

    Args:
        processed_red_dir (str): Caminho para as bandas vermelhas processadas.
        processed_green_dir (str): Caminho para as bandas verdes processadas.
        processed_blue_dir (str): Caminho para as bandas azuis processadas.
        processed_mask_dir (str): Caminho para as máscaras processadas.

    Returns:
        List[Tuple[str, str, str, str, str]]: Lista contendo tuplas das imagens encontradas.
    """
    red_files = sorted(os.listdir(processed_red_dir))
    green_files = sorted(os.listdir(processed_green_dir))
    blue_files = sorted(os.listdir(processed_blue_dir))
    mask_files = sorted(os.listdir(processed_mask_dir))

    def get_base_name(filename: str) -> str:
        base_name = os.path.splitext(filename)[0]
        prefixes = ['red_', 'green_', 'blue_', 'gt_']
        for prefix in prefixes:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        return base_name

    red_files_dict = {get_base_name(f): f for f in red_files}
    green_files_dict = {get_base_name(f): f for f in green_files}
    blue_files_dict = {get_base_name(f): f for f in blue_files}
    mask_files_dict = {get_base_name(f): f for f in mask_files}

    common_base_names = (set(red_files_dict.keys()) &
                         set(green_files_dict.keys()) &
                         set(blue_files_dict.keys()) &
                         set(mask_files_dict.keys()))

    print(f'Número de arquivos comuns: {len(common_base_names)}')

    file_pairs = []
    for base_name in sorted(common_base_names):
        r_file = red_files_dict[base_name]
        g_file = green_files_dict[base_name]
        b_file = blue_files_dict[base_name]
        m_file = mask_files_dict[base_name]
        file_pairs.append((base_name, r_file, g_file, b_file, m_file))

    return file_pairs


def is_band_all_zeros(band_path: str) -> bool:
    """
    Verifica se todos os pixels de uma banda são zero.

    Args:
        band_path (str): Caminho para o arquivo da banda.
    
    Returns:
        bool: True se todos os pixels forem zero, False caso contrário.
    """
    if not os.path.exists(band_path):
        return False
    try:
        with Image.open(band_path) as band:
            band_array = np.array(band)
        return np.all(band_array == 0)
    except Exception:
        # Caso não seja possível abrir a banda (arquivo corrompido ou inexistente)
        return False


def is_band_constant(band_path: str) -> bool:
    """
    Retorna True se a banda for completamente constante (todos os pixels
    têm o mesmo valor), seja preto, branco ou qualquer valor único.

    Args:
        band_path (str): Caminho para o arquivo da banda.

    Returns:
        bool: True se a banda for constante ou se não existir, False caso contrário.
    """
    if not os.path.exists(band_path):
        return True  # Se não existe, consideramos inválido/constante
    try:
        with Image.open(band_path) as band:
            band_array = np.array(band)
            # Verifica se todos os valores são iguais ao primeiro pixel
            return np.all(band_array == band_array.flat[0])
    except Exception:
        # Se der erro ao abrir a banda, consideramos que é inválido/constante
        return True


def are_bands_corrupted_or_mismatched(red_path: str,
                                      green_path: str,
                                      blue_path: str) -> bool:
    """
    Verifica se as bandas estão corrompidas (não abrem) ou com dimensões diferentes.
    Retorna True se houver qualquer problema.

    Args:
        red_path (str): Caminho da banda vermelha.
        green_path (str): Caminho da banda verde.
        blue_path (str): Caminho da banda azul.

    Returns:
        bool: True se as bandas estiverem corrompidas ou com formatos diferentes, False caso contrário.
    """
    try:
        with Image.open(red_path) as r_img:
            r_arr = np.array(r_img)
        with Image.open(green_path) as g_img:
            g_arr = np.array(g_img)
        with Image.open(blue_path) as b_img:
            b_arr = np.array(b_img)
    except Exception:
        # Caso alguma imagem não possa ser aberta, é corrompida
        return True

    # Verificar se dimensões batem
    if r_arr.shape != g_arr.shape or r_arr.shape != b_arr.shape:
        return True

    return False


def exclude_all(image_paths: List[str],
                mask_path: str,
                excluded_images_dir: str,
                excluded_masks_dir: str) -> None:
    """
    Move as imagens e a máscara para a pasta de excluídos.

    Args:
        image_paths (List[str]): Lista com caminhos das imagens a serem movidas.
        mask_path (str): Caminho da máscara a ser movida.
        excluded_images_dir (str): Caminho do diretório de imagens excluídas.
        excluded_masks_dir (str): Caminho do diretório de máscaras excluídas.
    """
    os.makedirs(excluded_images_dir, exist_ok=True)
    os.makedirs(excluded_masks_dir, exist_ok=True)

    for img_path in image_paths:
        if os.path.exists(img_path):
            shutil.move(img_path, os.path.join(excluded_images_dir, os.path.basename(img_path)))

    if os.path.exists(mask_path):
        shutil.move(mask_path, os.path.join(excluded_masks_dir, os.path.basename(mask_path)))


def process_cloud_data(
    file_pairs: List[Tuple[str, str, str, str, str]],
    train_red_dir: str,
    train_green_dir: str,
    train_blue_dir: str,
    train_gt_dir: str,
    processed_red_dir: str,
    processed_green_dir: str,
    processed_blue_dir: str,
    processed_masks_dir: str,
    excluded_images_dir: str,
    excluded_masks_dir: str,
    cloud_name: str = 'Cloud'
) -> List[Tuple[str, str, str, str, str]]:
    """
    Processa os pares de arquivos (R, G, B, Mascara):
    1. Verifica se todos os arquivos existem.
    2. Verifica se as bandas são corrompidas ou não batem em dimensões.
    3. Verifica se as bandas e a máscara estão totalmente zeradas.
    4. Copia para diretórios de destino se estiverem OK, caso contrário move para diretórios de excluídos.

    Args:
        file_pairs (List[Tuple[str, str, str, str, str]]): Lista de tuplas (base_name, r_file, g_file, b_file, m_file).
        train_red_dir (str): Diretório de bandas vermelhas originais.
        train_green_dir (str): Diretório de bandas verdes originais.
        train_blue_dir (str): Diretório de bandas azuis originais.
        train_gt_dir (str): Diretório de máscaras originais.
        processed_red_dir (str): Diretório de destino para bandas vermelhas processadas.
        processed_green_dir (str): Diretório de destino para bandas verdes processadas.
        processed_blue_dir (str): Diretório de destino para bandas azuis processadas.
        processed_masks_dir (str): Diretório de destino para máscaras processadas.
        excluded_images_dir (str): Diretório de destino para imagens excluídas.
        excluded_masks_dir (str): Diretório de destino para máscaras excluídas.
        cloud_name (str): Nome da nuvem (opcional, usado para descrição no tqdm).

    Returns:
        List[Tuple[str, str, str, str, str]]: Lista de tuplas dos arquivos que foram excluídos.
    """

    excluded_files = []

    for base_name, r_file, g_file, b_file, m_file in tqdm(file_pairs, desc=f'Processando imagens de {cloud_name}'):
        red_path = os.path.join(train_red_dir, r_file)
        green_path = os.path.join(train_green_dir, g_file)
        blue_path = os.path.join(train_blue_dir, b_file)
        mask_path = os.path.join(train_gt_dir, m_file)

        # Verificar se arquivos existem
        if not (os.path.exists(red_path) and os.path.exists(green_path) and 
                os.path.exists(blue_path) and os.path.exists(mask_path)):
            # Algum arquivo não existe, excluir
            exclude_all([red_path, green_path, blue_path], mask_path,
                        excluded_images_dir, excluded_masks_dir)
            excluded_files.append((base_name, r_file, g_file, b_file, m_file))
            continue

        # Verificar se as bandas são corrompidas ou tem tamanhos inconsistentes
        if are_bands_corrupted_or_mismatched(red_path, green_path, blue_path):
            # Arquivos problemáticos, mover para excluídos
            exclude_all([red_path, green_path, blue_path], mask_path,
                        excluded_images_dir, excluded_masks_dir)
            excluded_files.append((base_name, r_file, g_file, b_file, m_file))
            continue

        # Verificar se as bandas e a máscara são todas zero
        is_red_zero = is_band_all_zeros(red_path)
        is_green_zero = is_band_all_zeros(green_path)
        is_blue_zero = is_band_all_zeros(blue_path)

        mask_exists = os.path.exists(mask_path)
        if mask_exists:
            with Image.open(mask_path) as mask:
                mask_array = np.array(mask)
                is_zero_mask = np.all(mask_array == 0)
        else:
            is_zero_mask = True

        if (is_red_zero and is_green_zero and is_blue_zero and is_zero_mask):
            exclude_all([red_path, green_path, blue_path], mask_path,
                        excluded_images_dir, excluded_masks_dir)
            excluded_files.append((base_name, r_file, g_file, b_file, m_file))
            continue

        # Se chegou aqui, a imagem é utilizável, vamos copiar para os diretórios processados
        os.makedirs(processed_red_dir, exist_ok=True)
        os.makedirs(processed_green_dir, exist_ok=True)
        os.makedirs(processed_blue_dir, exist_ok=True)
        os.makedirs(processed_masks_dir, exist_ok=True)

        shutil.copy(red_path, os.path.join(processed_red_dir, r_file))
        shutil.copy(green_path, os.path.join(processed_green_dir, g_file))
        shutil.copy(blue_path, os.path.join(processed_blue_dir, b_file))
        shutil.copy(mask_path, os.path.join(processed_masks_dir, m_file))

    print(f'Processamento de {cloud_name} concluído. {len(excluded_files)} arquivos excluídos.')
    return excluded_files


def apply_clahe(image_array: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para melhorar o contraste.

    Args:
        image_array (np.ndarray): Array de imagem em tons de cinza (uint8).
        clip_limit (float): Limite de corte para o contraste.
        tile_grid_size (Tuple[int, int]): Tamanho do grid para o cálculo local do histograma.

    Returns:
        np.ndarray: Array resultante após aplicação de CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_array)


def load_and_normalize_band(path: str,
                            lower_percentile: float = 2.0,
                            upper_percentile: float = 98.0,
                            apply_hist_eq: bool = True) -> np.ndarray:
    """
    Carrega uma banda (imagem) e normaliza seus valores para [0, 255] usando percentis.

    Args:
        path (str): Caminho para a banda.
        lower_percentile (float): Percentil inferior para corte.
        upper_percentile (float): Percentil superior para corte.
        apply_hist_eq (bool): Aplica CLAHE para equalização de histograma se True.

    Returns:
        np.ndarray: Array (uint8) normalizado no intervalo [0, 255].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with Image.open(path) as img:
        arr = np.array(img).astype(np.float32)

    # Calcular percentis
    lower = np.percentile(arr, lower_percentile)
    upper = np.percentile(arr, upper_percentile)

    # Se upper == lower, evita divisão por zero (imagem constante)
    if upper == lower:
        # Pode-se colocar tudo como zero ou manter o array original sem normalizar
        arr_norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        # Cortar valores fora dos percentis
        arr = np.clip(arr, lower, upper)
        # Normalizar para [0,1]
        arr_norm = (arr - lower) / (upper - lower)
        # Garantir que não haja valores fora de [0,1] por qualquer imprecisão de float
        arr_norm = np.clip(arr_norm, 0.0, 1.0)
        # Converter para [0,255]
        arr_norm = (arr_norm * 255).astype(np.uint8)

    # Aplicar equalização de histograma (CLAHE) para corrigir iluminação e contraste
    if apply_hist_eq:
        arr_norm = apply_clahe(arr_norm)

    return arr_norm


def load_and_process_mask(path: str) -> np.ndarray:
    """
    Carrega a máscara e a converte para binário (255 para pixels > 0, senão 0).

    Args:
        path (str): Caminho para a máscara.

    Returns:
        np.ndarray: Máscara binária (uint8) com valores 0 ou 255.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo da máscara não encontrado: {path}")

    with Image.open(path) as img:
        mask = img.convert('L')
        mask_array = np.array(mask)
        mask_binary = (mask_array > 0).astype(np.uint8) * 255
    return mask_binary


def combine_rgb(red: np.ndarray,
                green: np.ndarray,
                blue: np.ndarray) -> np.ndarray:
    """
    Combina três bandas em um array RGB.

    Args:
        red (np.ndarray): Banda vermelha.
        green (np.ndarray): Banda verde.
        blue (np.ndarray): Banda azul.

    Returns:
        np.ndarray: Imagem RGB combinada.
    """
    if red.shape != green.shape or red.shape != blue.shape:
        raise ValueError("As bandas R, G e B devem ter a mesma forma")
    rgb = np.dstack((red, green, blue))
    return rgb


def visualize_rgb_and_mask(file_pairs: List[Tuple[str, str, str, str, str]],
                           train_red_dir: str,
                           train_green_dir: str,
                           train_blue_dir: str,
                           train_gt_dir: str,
                           images_per_figure: int = 5) -> None:
    """
    Visualiza lado a lado a imagem RGB e a máscara, em lotes de 'images_per_figure'.

    Args:
        file_pairs (List[Tuple[str, str, str, str, str]]): Lista de tuplas (base_name, r_file, g_file, b_file, m_file).
        train_red_dir (str): Diretório de bandas vermelhas originais.
        train_green_dir (str): Diretório de bandas verdes originais.
        train_blue_dir (str): Diretório de bandas azuis originais.
        train_gt_dir (str): Diretório de máscaras originais.
        images_per_figure (int): Quantidade de imagens por figura.
    """
    total_images = len(file_pairs)
    num_figures = (total_images + images_per_figure - 1) // images_per_figure

    for fig_num in range(num_figures):
        plt.figure(figsize=(15, 5 * images_per_figure))
        start_idx = fig_num * images_per_figure
        end_idx = min(start_idx + images_per_figure, total_images)

        for i, (base_name, r_file, g_file, b_file, m_file) in enumerate(file_pairs[start_idx:end_idx], 1):
            red_path = os.path.join(train_red_dir, r_file)
            green_path = os.path.join(train_green_dir, g_file)
            blue_path = os.path.join(train_blue_dir, b_file)
            mask_path = os.path.join(train_gt_dir, m_file)

            red_band = load_and_normalize_band(red_path)
            green_band = load_and_normalize_band(green_path)
            blue_band = load_and_normalize_band(blue_path)
            mask = load_and_process_mask(mask_path)

            rgb_image = combine_rgb(red_band, green_band, blue_band)

            plt.subplot(images_per_figure, 2, (i - 1) * 2 + 1)
            plt.imshow(rgb_image)
            plt.title(f'{base_name} - RGB')
            plt.axis('off')

            plt.subplot(images_per_figure, 2, (i - 1) * 2 + 2)
            plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
            plt.title(f'{base_name} - Máscara')
            plt.axis('off')

        plt.tight_layout()
        plt.show()


def identify_low_information_images(processed_red_dir: str,
                                   processed_green_dir: str,
                                   processed_blue_dir: str,
                                   processed_masks_dir: str,
                                   threshold_percentage: float = 10.0) -> List[Tuple[str, str, str, str, str]]:
    """
    Identifica imagens com pouca informação com base na percentagem de pixels não nulos.

    Args:
        processed_red_dir (str): Diretório das bandas vermelha processadas.
        processed_green_dir (str): Diretório das bandas verde processadas.
        processed_blue_dir (str): Diretório das bandas azul processadas.
        processed_masks_dir (str): Diretório das máscaras processadas.
        threshold_percentage (float): Percentagem mínima de pixels não nulos para considerar uma imagem como válida.

    Returns:
        List[Tuple[str, str, str, str, str]]: Lista de tuplas contendo (base_name, r_file, g_file, b_file, m_file).
    """
    # Obter base_names comuns
    file_pairs = create_file_pairs(processed_red_dir, processed_green_dir, processed_blue_dir, processed_masks_dir)
    low_info_file_pairs = []

    for base_name, r_file, g_file, b_file, m_file in tqdm(file_pairs, desc='Identificando imagens com pouca informação'):
        r_path = os.path.join(processed_red_dir, r_file)
        g_path = os.path.join(processed_green_dir, g_file)
        b_path = os.path.join(processed_blue_dir, b_file)
        m_path = os.path.join(processed_masks_dir, m_file)

        # Carregar as bandas
        try:
            red_band = np.array(Image.open(r_path))
            green_band = np.array(Image.open(g_path))
            blue_band = np.array(Image.open(b_path))
        except Exception as e:
            print(f"Erro ao abrir as bandas para {base_name}: {e}")
            continue

        # Calcular o número total de pixels
        total_pixels = red_band.size

        # Calcular o número de pixels não nulos em cada banda
        non_zero_r = np.count_nonzero(red_band)
        non_zero_g = np.count_nonzero(green_band)
        non_zero_b = np.count_nonzero(blue_band)

        # Calcular a percentagem de pixels não nulos combinados
        combined_non_zero = non_zero_r + non_zero_g + non_zero_b
        combined_total = total_pixels * 3  # Três bandas

        percent_non_zero = (combined_non_zero / combined_total) * 100

        if percent_non_zero < threshold_percentage:
            low_info_file_pairs.append((base_name, r_file, g_file, b_file, m_file))

    print(f'Total de imagens com pouca informação: {len(low_info_file_pairs)}')
    return low_info_file_pairs


def visualize_and_manage_images_masks(images_dir: str = '../data/dados/images',
                                      masks_dir: str = '../data/dados/masks') -> None:
    """
    Seleciona aleatoriamente uma imagem e sua respectiva máscara, visualiza-as e permite ao usuário
    excluir os arquivos se desejado. A visualização anterior é limpa antes de exibir a próxima.

    Args:
        images_dir (str): Caminho para o diretório que contém as imagens.
        masks_dir (str): Caminho para o diretório que contém as máscaras.
    """
    # Verifica se os diretórios existem
    if not os.path.isdir(images_dir):
        print(f"O diretório de imagens '{images_dir}' não existe.")
        return
    if not os.path.isdir(masks_dir):
        print(f"O diretório de máscaras '{masks_dir}' não existe.")
        return

    # Lista todos os arquivos PNG no diretório de imagens
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

    if not image_files:
        print("Nenhuma imagem encontrada no diretório especificado.")
        return

    while True:
        # Seleciona uma imagem aleatória
        random_image = random.choice(image_files)
        # Extrai o ID único do nome do arquivo (assumindo que o nome do arquivo é '<unique_id>.png')
        unique_id = os.path.splitext(random_image)[0]

        # Define o caminho para a máscara correspondente
        mask_file = f"{unique_id}.png"
        mask_path = os.path.join(masks_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Máscara correspondente para a imagem '{random_image}' não foi encontrada.")
            # Remove a imagem da lista para evitar tentativas futuras
            image_files.remove(random_image)
            if not image_files:
                print("Nenhuma imagem válida restante para visualizar.")
                break
            continue

        # Carrega a imagem e a máscara
        try:
            with Image.open(os.path.join(images_dir, random_image)) as im:
                image = np.array(im)
        except Exception as e:
            print(f"Erro ao abrir a imagem '{random_image}': {e}")
            # Remove a imagem problemática da lista
            image_files.remove(random_image)
            if not image_files:
                print("Nenhuma imagem válida restante para visualizar.")
                break
            continue

        try:
            with Image.open(mask_path) as m:
                mask = np.array(m)
        except Exception as e:
            print(f"Erro ao abrir a máscara '{mask_file}': {e}")
            # Remove a máscara problemática da lista
            image_files.remove(random_image)
            if not image_files:
                print("Nenhuma imagem válida restante para visualizar.")
                break
            continue

        # Verifica se a imagem e a máscara têm as mesmas dimensões
        if image.shape[:2] != mask.shape[:2]:
            print(f"A imagem '{random_image}' e a máscara '{mask_file}' têm dimensões diferentes.")
            # Remove a imagem da lista para evitar problemas futuros
            image_files.remove(random_image)
            if not image_files:
                print("Nenhuma imagem válida restante para visualizar.")
                break
            continue

        # Limpa a saída anterior
        clear_output(wait=True)

        # Exibe a imagem e a máscara lado a lado
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Imagem RGB')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Máscara')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Pergunta ao usuário se deseja excluir os arquivos
        while True:
            user_input = input(f"Deseja excluir a imagem '{random_image}' e sua máscara '{mask_file}'? (s/n): ").strip().lower()
            if user_input == 's':
                try:
                    os.remove(os.path.join(images_dir, random_image))
                    os.remove(mask_path)
                    print(f"Imagem '{random_image}' e máscara '{mask_file}' foram excluídas com sucesso.")
                    # Remove a imagem da lista após exclusão
                    image_files.remove(random_image)
                except Exception as e:
                    print(f"Erro ao excluir os arquivos: {e}")
                break
            elif user_input == 'n':
                print("Arquivos mantidos.")
                break
            else:
                print("Entrada inválida. Por favor, responda com 's' para sim ou 'n' para não.")

        # Pergunta ao usuário se deseja visualizar outra imagem
        while True:
            continuar = input("Deseja visualizar outra imagem? (s/n): ").strip().lower()
            if continuar == 's':
                break
            elif continuar == 'n':
                print("Finalizando a visualização.")
                return
            else:
                print("Entrada inválida. Por favor, responda com 's' para sim ou 'n' para não.")



# Função para normalizar a imagem original para visualização
def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """
    Normaliza um array de imagem para o intervalo [0, 255] como uint8 para visualização.
    """
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    else:
        arr_normalized = (arr - arr_min) / (arr_max - arr_min)  # Escala para [0, 1]
        arr_normalized = (arr_normalized * 255).astype(np.uint8)  # Escala para [0, 255]
        return arr_normalized
    

# Função para visualizar antes e depois
def visualizar_rgb_normalizacao(red_path: str, green_path: str, blue_path: str):
    # Carregar as bandas originais
    with Image.open(red_path) as img:
        red_original = np.array(img)
    with Image.open(green_path) as img:
        green_original = np.array(img)
    with Image.open(blue_path) as img:
        blue_original = np.array(img)

    # Normalizar as bandas originais para visualização
    red_display = normalize_for_display(red_original)
    green_display = normalize_for_display(green_original)
    blue_display = normalize_for_display(blue_original)

    # Combinar as bandas originais normalizadas em RGB
    rgb_original = combine_rgb(red_display, green_display, blue_display)

    # Normalizar cada banda usando a função definida
    red_normalizado = load_and_normalize_band(red_path)
    green_normalizado = load_and_normalize_band(green_path)
    blue_normalizado = load_and_normalize_band(blue_path)

    # Combinar as bandas normalizadas em RGB
    rgb_normalizado = combine_rgb(red_normalizado, green_normalizado, blue_normalizado)

    # Plotar as imagens
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_original)
    plt.title('Antes da Normalização')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_normalizado)
    plt.title('Depois da Normalização')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
