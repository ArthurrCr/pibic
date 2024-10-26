import json

def json_to_mlt(json_file, mlt_file):
    # Ler o arquivo JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Abrir o arquivo MLT para escrita
    with open(mlt_file, 'w') as f:
        # Escrever os metadados principais
        f.write(f"Type: {data['type']}\n")
        f.write(f"Version: {data['version']}\n")
        f.write(f"ID: {data['id']}\n")
        f.write(f"Image Quality: {data['properties'].get('IMAGE_QUALITY', 'N/A')}\n")
        f.write(f"Date Acquired: {data['properties'].get('DATE_ACQUIRED', 'N/A')}\n")
        f.write(f"Cloud Cover: {data['properties'].get('CLOUD_COVER', 'N/A')}\n")
        f.write(f"Sun Azimuth: {data['properties'].get('SUN_AZIMUTH', 'N/A')}\n")
        f.write(f"Sun Elevation: {data['properties'].get('SUN_ELEVATION', 'N/A')}\n")
        f.write("Bands:\n")

        # Escrever os detalhes das bandas
        for band in data['bands']:
            f.write(f"  Band ID: {band['id']}\n")
            f.write(f"    Data Type: {band['data_type']['type']}\n")
            f.write(f"    Precision: {band['data_type']['precision']}\n")
            f.write(f"    Min: {band['data_type']['min']}\n")
            f.write(f"    Max: {band['data_type']['max']}\n")
            f.write(f"    Dimensions: {band['dimensions'][0]} x {band['dimensions'][1]}\n")
            f.write(f"    CRS: {band['crs']}\n")
            f.write(f"    CRS Transform: {band['crs_transform']}\n")

        f.write("Properties:\n")
        for key, value in data['properties'].items():
            f.write(f"  {key}: {value}\n")
    
    print(f"Conversion complete. MLT file saved as {mlt_file}")

# Uso da função
json_to_mlt('/media/weverton/D/Remote Sensing/fmask_simple_implementation/Seixas/6B/2004/seixas_20040905.json', 'metadados.mlt')
