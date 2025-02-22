import re
import ast
import rasterio
from pyproj import Transformer
from geemap import ee_to_numpy
import numpy as np
import ee
from shapely import wkt
import matplotlib.pyplot as plt
import json

def parse_numpy_repr(array_val):
    """
    Converte uma representação de array (string, list ou np.ndarray)
    para uma lista de valores numéricos.
    """
    if isinstance(array_val, np.ndarray):
        return array_val.tolist()
    if isinstance(array_val, (list, tuple)):
        return array_val
    if isinstance(array_val, str):
        s_no_dtype = re.sub(r",\s*dtype=[^)]*\)", ")", array_val.strip())
        s_clean = re.sub(r"^array\((.*)\)$", r"\1", s_no_dtype)
        return ast.literal_eval(s_clean)
    return array_val

def get_utm_bounds(metadata_sample):
    geotransform = parse_numpy_repr(metadata_sample["stac:geotransform"])
    width_height = parse_numpy_repr(metadata_sample["stac:raster_shape"])
    x_min = geotransform[0]
    y_max = geotransform[3]
    dx    = geotransform[1]
    dy    = geotransform[5]
    width, height = width_height
    x_max = x_min + (width * dx)
    y_min = y_max + (height * dy)
    return (x_min, y_min, x_max, y_max)

def utm_to_wgs84(bounds_utm, crs_utm):
    crs_utm = f"EPSG:{crs_utm}"
    transformer = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
    x_min_wgs84, y_min_wgs84 = transformer.transform(bounds_utm[0], bounds_utm[1])
    x_max_wgs84, y_max_wgs84 = transformer.transform(bounds_utm[2], bounds_utm[3])
    return ee.Geometry.BBox(x_min_wgs84, y_min_wgs84, x_max_wgs84, y_max_wgs84)

def get_s2_sr_cld_col(s2_id, metadata_sample):
    bounds_utm = get_utm_bounds(metadata_sample)
    crs_utm = metadata_sample["stac:crs"].split(":")[1]
    region = utm_to_wgs84(bounds_utm, crs_utm)
    s2_sr_img = ee.Image("COPERNICUS/S2_SR/" + s2_id).clip(region)
    s2_cloud_img = ee.Image("COPERNICUS/S2_CLOUD_PROBABILITY/" + s2_id).clip(region)
    return s2_sr_img.set('s2cloudless', s2_cloud_img)

def plot_rgb_comparison(sample_index, test_tdf, parts, tacoreader, candidate_s2_id_gee):
    """
    Plota lado a lado:
      - A imagem RGB local (do dataset)
      - A imagem RGB do Sentinel-2 do Earth Engine,
    utilizando o candidato informado para s2_id_gee, se fornecido.
    """
    ds = tacoreader.load(parts)
    sample = ds.read(sample_index)

    with rasterio.open(sample.read(0)) as src:
        bands = src.read(range(1, src.count + 1))
    bands = bands.transpose(1, 2, 0)

    rgb_local = bands[:, :, [3, 2, 1]].astype(np.float32)
    perc_local = np.percentile(rgb_local, 98)
    rgb_local = np.clip(rgb_local / perc_local, 0, 1)

    # Copia os metadados da amostra e sobrescreve o s2_id_gee, se houver candidato
    metadata_sample = test_tdf.iloc[sample_index].copy()
    metadata_sample["s2_id_gee"] = candidate_s2_id_gee

    s2_id = metadata_sample["s2_id"]
    s2_id_gee = metadata_sample["s2_id_gee"]
    centroid_wkt = metadata_sample["stac:centroid"]

    s2_img = get_s2_sr_cld_col(s2_id_gee, metadata_sample)

    patch_size = 512
    scale = 10
    centroid = wkt.loads(centroid_wkt)
    lon, lat = centroid.x, centroid.y
    ee_point = ee.Geometry.Point([lon, lat])
    buffer = ee_point.buffer((patch_size * scale) / 2)
    region = buffer.bounds()

    rgb_ee = s2_img.select(['B4', 'B3', 'B2'])
    rgb_ee_arr = ee_to_numpy(rgb_ee, region=region, scale=scale)
    if rgb_ee_arr is None:
        raise ValueError("Earth Engine retornou nenhum dado para essa região. Verifique a região ou o ID.")

    # Normaliza a imagem EE para visualização
    rgb_ee_norm = rgb_ee_arr.astype(np.float32)
    perc_ee = np.percentile(rgb_ee_norm, 98)
    rgb_ee_norm = np.clip(rgb_ee_norm / perc_ee, 0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(rgb_local)
    ax[0].set_title(f"Local RGB (s2_id={s2_id})")
    ax[0].axis("off")

    ax[1].imshow(rgb_ee_norm)
    ax[1].set_title(f"EE Sentinel-2 RGB (s2_id_gee={s2_id_gee})")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
