import os
from huggingface_hub import hf_hub_download

def download_cloudsen12(local_dir="./data/dados", type =""):
    """
    Baixa as partes do CloudSEN12+ (modo L1C) para o diretório especificado.
    Se a pasta não existir, ela será criada.
    
    Parâmetros:
      - local_dir: diretório onde salvar os arquivos.
      - token: token de autenticação do Hugging Face (se necessário).
    
    Retorna:
      Uma lista com os caminhos dos arquivos baixados.
    """
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    if type == "L1C":
      part1 = hf_hub_download(
          repo_id="tacofoundation/CloudSEN12", 
          filename="cloudsen12-l1c.0000.part.taco", 
          repo_type="dataset", 
          local_dir=local_dir
      )
      #part2 = hf_hub_download(
      #    repo_id="tacofoundation/CloudSEN12", 
      #    filename="cloudsen12-l1c.0001.part.taco", 
      #    repo_type="dataset", 
      #    local_dir=local_dir
      #)
      #part3 = hf_hub_download(
      #    repo_id="tacofoundation/CloudSEN12", 
      #    filename="cloudsen12-l1c.0002.part.taco", 
      #    repo_type="dataset", 
      #    local_dir=local_dir
      #)
      #part4 = hf_hub_download(
      #    repo_id="tacofoundation/CloudSEN12", 
      #    filename="cloudsen12-l1c.0003.part.taco", 
      #    repo_type="dataset", 
      #    local_dir=local_dir
      #)
      part5 = hf_hub_download(
          repo_id="tacofoundation/CloudSEN12", 
          filename="cloudsen12-l1c.0004.part.taco", 
          repo_type="dataset", 
          local_dir=local_dir
      )
    elif type == "L2A":
      part1 = hf_hub_download(
          repo_id="tacofoundation/CloudSEN12", 
          filename="cloudsen12-l2a.0000.part.taco", 
          repo_type="dataset", 
          local_dir=local_dir
      )
      #part2 = hf_hub_download(
      #    repo_id="tacofoundation/CloudSEN12", 
      #    filename="cloudsen12-l2a.0001.part.taco", 
      #    repo_type="dataset", 
      #    local_dir=local_dir
      #)
      #part3 = hf_hub_download(
      #    repo_id="tacofoundation/CloudSEN12", 
      #    filename="cloudsen12-l2a.0002.part.taco", 
      #    repo_type="dataset", 
      #    local_dir=local_dir
      #)
      #part4 = hf_hub_download(
      #    repo_id="tacofoundation/CloudSEN12", 
      #    filename="cloudsen12-l2a.0003.part.taco", 
      #    repo_type="dataset", 
      #    local_dir=local_dir
      #)
      part5 = hf_hub_download(
          repo_id="tacofoundation/CloudSEN12", 
          filename="cloudsen12-l2a.0004.part.taco", 
          repo_type="dataset", 
          local_dir=local_dir
      )
    elif type == "extra":
      part1 = hf_hub_download(
          repo_id="tacofoundation/CloudSEN12", 
          filename="cloudsen12-extra.0000.part.taco", 
          repo_type="dataset", 
          local_dir=local_dir
      )

    return [part1,part5] # para evitar ter que baixar tudo, baixa só as partes que estão a label high

