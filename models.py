import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)     
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        

        out = self.gamma * out + x
        return out


class Generator(nn.Module):

    def __init__(self, noise_dim=100, embedding_dim=512, channels=3):
        super(Generator, self).__init__()
        input_dim = noise_dim + embedding_dim

        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )

        def upsample_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

        self.main = nn.Sequential(
            upsample_block(1024, 512),  # Output: 512 x 8 x 8
            upsample_block(512, 256),   # Output: 256 x 16 x 16
            upsample_block(256, 128),   # Output: 128 x 32 x 32
            SelfAttention(128),         # SELF-ATTENTION
            upsample_block(128, 64),    # Output: 64 x 64 x 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, embedding):
        combined_input = torch.cat([noise, embedding], dim=1)
        reshaped_input = combined_input.view(-1, combined_input.size(1), 1, 1)
        out = self.init_block(reshaped_input)
        out = self.main(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, embedding_dim=512, channels=3):
        super(Discriminator, self).__init__()
        
        self.image_path = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            SelfAttention(256),         # SELF-ATTENTION
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.combined_path = nn.Sequential(
            nn.Conv2d(512 + embedding_dim, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image, embedding):
        image_features = self.image_path(image)
        embedding_reshaped = embedding.view(-1, embedding.size(1), 1, 1)
        embedding_expanded = embedding_reshaped.expand(-1, -1, image_features.size(2), image_features.size(3))
        combined = torch.cat([image_features, embedding_expanded], dim=1)
        output = self.combined_path(combined)
        return output.view(-1, 1).squeeze(1)

