"""
Architettura ResNet per la classificazione di immagini SVHN.
La ResNet utilizza blocchi residui per migliorare l'apprendimento e evitare il vanishing gradient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResidualBlock(nn.Module):
    """
    Blocco residuo - componente fondamentale della ResNet.
    Permette ai dati di "saltare" alcuni strati (skip connection).
    Questo aiuta a evitare il problema del vanishing gradient.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Primo strato convoluzionale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Secondo strato convoluzionale
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut (skip connection): permette il salto diretto dell'input
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Se le dimensioni cambiano, adatta l'input con una convoluzione 1x1
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass: applica le convoluzioni e somma con il percorso residuo (shortcut).
        """
        # Primo blocco convoluzionale con ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Secondo blocco convoluzionale senza attivazione
        out = self.bn2(self.conv2(out))
        # Somma il percorso residuo (skip connection) - questo è il cuore della ResNet!
        out += self.shortcut(x)
        # Applica ReLU al risultato finale
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
    Architettura ResNet-31 per la classificazione di 10 classi (numeri 0-9).
    Struttura:
    - 1 strato convoluzionale iniziale
    - 3 serie di blocchi residui (layer1, layer2, layer3)
    - Average pooling e fully connected layer per la classificazione finale
    """
    
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        # Numero di canali nel primo strato
        self.in_channels = 64

        # Strato convoluzionale iniziale: 3 canali (RGB) -> 64 canali
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Tre serie di strati con blocchi residui
        # Layer1: 64 canali (stride=1, no dimensione reduction)
        self.layer1 = self.make_layer(64, 2, stride=1)
        # Layer2: 128 canali (stride=2, riduce la dimensione spaziale a metà)
        self.layer2 = self.make_layer(128, 2, stride=2)
        # Layer3: 256 canali (stride=2, riduce ancora la dimensione spaziale)
        self.layer3 = self.make_layer(256, 2, stride=2)
        
        # Average pooling adattivo: riduce qualsiasi input a 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer finale: 256 -> num_classes (10)
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        """
        Crea una serie di blocchi residui.
        Args:
            out_channels: numero di canali di output
            num_blocks: numero di blocchi residui in questa serie
            stride: stride del primo blocco (gli altri hanno stride=1)
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass attraverso l'intera rete.
        """
        # Strato convoluzionale iniziale
        out = F.relu(self.bn1(self.conv1(x)))
        # Tre serie di blocchi residui
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # Average pooling: riduce a 1x1 (mantenendo i 256 canali)
        out = self.avg_pool(out)
        # Appiattisce il tensore per il fully connected layer
        out = out.view(out.size(0), -1)
        # Fully connected layer: produce le probabilità per le 10 classi
        out = self.fc(out)
        return out

class ResNetTransfer(nn.Module):
    """
    Transfer Learning ResNet50 pretrained su ImageNet.
    Utilizza i pesi pretrained e modifica l'ultimo layer per la classificazione SVHN (10 classi).
    Permette di freezare i pesi dei layer precedenti per un fine-tuning veloce.
    """
    
    def __init__(self, num_classes=10, freeze_backbone=True):
        """
        Args:
            num_classes: numero di classi (10 per SVHN)
            freeze_backbone: se True, congela i pesi del backbone pretrained
        """
        super(ResNetTransfer, self).__init__()
        
        # Carica ResNet50 pretrained su ImageNet
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Congela i pesi del backbone se richiesto (fine-tuning veloce)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Backbone ResNet50 congelato - Solo il layer finale sarà addestrato")
        else:
            print("Backbone ResNet50 scongelato - Tutti i layer saranno addestrati (fine-tuning completo)")
        
        # Sostituisci l'ultimo fully connected layer
        # ResNet50 ha 2048 feature in output prima dell'FC layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Sblocca il layer finale per l'addestramento
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass attraverso la ResNet50.
        """
        return self.model(x)