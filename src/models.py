import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, model_scale=0.75):
        super(Generator, self).__init__()
        f = [int(s * model_scale) for s in [64, 128, 256, 512, 512, 512, 512, 512]]
        
        # Encoder
        self.down1 = self._downsample(3, f[0], apply_batchnorm=False)
        self.down2 = self._downsample(f[0], f[1])
        self.down3 = self._downsample(f[1], f[2])
        self.down4 = self._downsample(f[2], f[3])
        self.down5 = self._downsample(f[3], f[4])
        self.down6 = self._downsample(f[4], f[5])
        self.down7 = self._downsample(f[5], f[6])
        self.down8 = self._downsample(f[6], f[7])
        
        # Decoder
        self.up1 = self._upsample(f[7], f[6], apply_dropout=True)
        self.up2 = self._upsample(f[6]*2, f[5], apply_dropout=True)
        self.up3 = self._upsample(f[5]*2, f[4], apply_dropout=True)
        self.up4 = self._upsample(f[4]*2, f[3])
        self.up5 = self._upsample(f[3]*2, f[2])
        self.up6 = self._upsample(f[2]*2, f[1])
        self.up7 = self._upsample(f[1]*2, f[0])
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(f[0]*2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    # fonction de downsample pour l'encodeur
    def _downsample(self, in_channels, out_channels, apply_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    # fonction d'upsample pour le décodeur
    def _upsample(self, in_channels, out_channels, apply_dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    # fonction de forward propagation
    def forward(self, x):
        # Résultat du passage par chaques couches de l'encodeur
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Résultat du pasage par chaques couches du décodeur avec des connexions de saut
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        
        # Résultat final après concaténation avec la première couche de l'encodeur
        return self.final(torch.cat([u7, d1], dim=1))

# le discriminator model est celui qui va vérifier si l'image proposé est une vrai image ou une fausse
class Discriminator(nn.Module):
    def __init__(self, model_scale=0.75):
        super(Discriminator, self).__init__()
        f = [int(s * model_scale) for s in [64, 128, 256, 512]]
        
        # couches de downsample pour le discriminateur
        self.down1 = self._downsample(6, f[0], apply_batchnorm=False)
        self.down2 = self._downsample(f[0], f[1])
        self.down3 = self._downsample(f[1], f[2])
        
        # couche finale du discriminateur
        self.conv = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(f[2], f[3], kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(f[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(f[3], 1, kernel_size=4, stride=1)
        )
        
        self._init_weights()
    
    # fonction de downsample pour le discriminateur
    def _downsample(self, in_channels, out_channels, apply_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    # fonction de forward propagation
    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        # Passage par les couches de downsample
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        # Passage par la couche finale
        return self.conv(x)
