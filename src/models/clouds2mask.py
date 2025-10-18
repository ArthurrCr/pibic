import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CloudS2Mask(nn.Module):
   
    def __init__(self,
                 encoder_name='tu-regnetz_d8',  # Prefixo 'tu-' para o encoder
                 encoder_weights=None,  
                 in_channels=13,
                 classes=4,
                 freeze_encoder=False):
     
        super(CloudS2Mask, self).__init__()

       
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        
        if freeze_encoder:
            for param in self.unet.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        return self.unet(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
       
        self.eval()
        logits = self.forward(x)                  
        probs = torch.softmax(logits, dim=1)      
        labels = probs.argmax(dim=1)             
        return labels
