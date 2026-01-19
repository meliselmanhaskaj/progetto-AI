import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def plot_class_distribution(path):
    # Caricamento del dataset SVHN
    data = sio.loadmat(path)

    # Estrazione e correzione delle etichette
    labels = data['y'].flatten()
    labels[labels == 10] = 0  # Corregge la classe '10' a '0'

    # Conteggio delle classi
    class_counts = Counter(labels)

    # Preparazione dei dati per il grafico
    classes = np.arange(10)
    counts = [class_counts[i] for i in classes]

    # Creazione del grafico
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xlabel('Classi (0-9)')
    plt.ylabel('Numero di immagini')
    plt.title('Distribuzione delle Classi nel Dataset SVHN')
    plt.xticks(classes)
    
    # Creazione cartella out se non esiste
    import os
    os.makedirs('out', exist_ok=True)
    plt.savefig('out/distribuzione_classi_svhn.png')
    
    # Mostra il grafico
    plt.show()
    
    # Chiudi la figura dopo averla mostrata
    plt.close()  # Chiudiamo esplicitamente la figura
    
    # Stampa dei valori numerici
    for i in range(10):
        print(f"Classe {i}: {class_counts[i]} immagini")