"""
Architettura ResNet per la classificazione di immagini SVHN.
La ResNet utilizza blocchi residui per migliorare l'apprendimento e evitare il vanishing gradient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
