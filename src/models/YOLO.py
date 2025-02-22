import torch
import torch.nn as nn
from ultralytics import YOLO

class CloudYOLO(nn.Module):
    def __init__(self, model_variant='yolov8l-seg', freeze_backbone=False, device='cuda'):
        """
        Inicializa o modelo YOLOv8 para segmentação de nuvens.
        
        Parâmetros:
          - model_variant (str): variante do modelo YOLOv8 de segmentação a ser usada 
                                  (ex.: 'yolov8s-seg', 'yolov8m-seg', etc.).  
          - freeze_backbone (bool): se True, congela os parâmetros da backbone.
          - device (str): dispositivo para executar o modelo ('cuda' ou 'cpu').
        """
        super().__init__()
        self.device = device
        # Carrega o modelo YOLOv8 de segmentação a partir dos pesos pré-treinados
        self.yolo = YOLO(f'{model_variant}.pt')
        self.yolo.model.to(self.device)
        
        if freeze_backbone:
            # Congela os parâmetros da backbone para treinamento somente dos cabeçalhos
            for param in self.yolo.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Passagem forward do modelo.
        
        Parâmetros:
          - x (Tensor): imagens de entrada, formato (B, 3, H, W).
        
        Retorno:
          - Durante o treinamento, o YOLOv8 calcula as predições e a loss (dependendo da configuração).
            Aqui, simplesmente chamamos o forward do modelo YOLO.  
        
        Atenção:
          O método forward do modelo YOLOv8 pode retornar estruturas de dados complexas (por exemplo,
          um dicionário com predições e loss). Se você for utilizar uma função de treinamento customizada,
          certifique-se de ajustar o processamento dos outputs conforme a documentação da sua versão.
        """
        return self.yolo.model(x)

    @torch.no_grad()
    def predict(self, x, conf=0.5):
        """
        Realiza a inferência no modo de predição.
        
        Parâmetros:
          - x (Tensor ou lista de imagens): as imagens de entrada.
          - conf (float): limiar de confiança para filtrar predições.
        
        Retorno:
          - results: uma lista de objetos de resultados (ultralytics.yolo.engine.results.Results),
            onde cada objeto contém as predições, incluindo as máscaras de segmentação (em .masks).
        """
        self.yolo.model.eval()
        results = self.yolo(x, augment=False, conf=conf)
        return results