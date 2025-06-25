import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceNetEncoder(nn.Module):
    def __init__(self, device):
        super(FaceNetEncoder, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2').to(device)
        self.model.eval() 

    def forward(self, image_batch):
        embeddings = self.model(image_batch)
        return embeddings

