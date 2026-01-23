"""
Script principale per il training e testing della ResNet su dataset SVHN.
Carica la configurazione da config.json e avvia il processo di apprendimento.
"""

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import jsonschema
import json
import sys
import numpy as np
import random
from pathlib import Path
from types import SimpleNamespace
from analysis import plot_class_distribution
from net_runner import NetRunner

def set_seeds(seed):
    """
    Imposta i seed per tutte le librerie di randomicità per garantire riproducibilità.
    Args:
        seed: valore del seed (es. 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Abilita la riproducibilità di CUDA (può essere più lento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds impostati a {seed} per riproducibilità")

def ifDataExist(directory):
    """
    Verifica se il dataset SVHN è già stato scaricato.
    Ritorna True se bisogna scaricarlo, False se è già presente.
    """
    data_path = Path(directory)
    if not data_path.exists() or not data_path.is_dir():
        print(f"La directory {directory} non esiste o non è valida.")
        return True
    
    # Controlla se ci sono almeno 2 file .mat (train e test)
    mat_files = list(data_path.glob('*.mat'))
    
    if len(mat_files) >= 2:
        print(f"Sono stati trovati {len(mat_files)} file .mat.")
        return False
    else:
        print(f"Sono stati trovati solo {len(mat_files)} file .mat.")
        return True
    
def main(cfg):
    """
    Funzione principale che configura il training/testing.
    Carica i dati, applica le trasformazioni e avvia NetRunner.
    """
    
    # Imposta i seed IMMEDIATAMENTE all'inizio per garantire riproducibilità
    # Questo deve essere la PRIMA operazione per evitare qualsiasi fonte di randomicità
    set_seeds(cfg.config.seed)
    
    # Crea un generator con seed per i DataLoader (garantisce riproducibilità dello shuffle)
    generator = torch.Generator()
    generator.manual_seed(cfg.config.seed)
    
    # Verifica se il dataset esiste, altrimenti lo scarica
    DOWNLOAD = ifDataExist('./data')
    
    # Crea la directory di output se non esiste
    out_path = Path(cfg.io.out_path)
    if not out_path.exists():
        out_path.mkdir()
    
    # Definisce le trasformazioni dei dati (Data Augmentation):
    # - Rotazione casuale fino a 10 gradi (aumenta variabilità)
    # - Conversione a tensore
    # - Normalizzazione per migliorare il training
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Data Augmentation: rotazione casuale
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalizza i valori RGB
    ])

    # Caricamento del dataset SVHN (Street View House Numbers)
    train_set = torchvision.datasets.SVHN(root='./data', split='train', download=DOWNLOAD, transform=transform)
    test_set = torchvision.datasets.SVHN(root='./data', split='test', download=DOWNLOAD, transform=transform)
     
    # Mostra il grafico della distribuzione delle classi
    print("Analizzando la distribuzione delle classi...")
    plot_class_distribution('./data/train_32x32.mat')

    # Riduce il dataset se configurato (utile per test rapidi)
    if cfg.config.reduce_dataset:
        print(f"Riducendo il dataset al {cfg.config.reduction_factor*100}%...")
        train_indices, _ = train_test_split(range(len(train_set)), train_size=cfg.config.reduction_factor, stratify=train_set.labels, random_state=cfg.config.seed)
        test_indices, _ = train_test_split(range(len(test_set)), train_size=cfg.config.reduction_factor, stratify=test_set.labels, random_state=cfg.config.seed)
        
        train_set = Subset(train_set, train_indices)
        test_set = Subset(test_set, test_indices)

    # Divide il training set in training (80%) e validazione (20%)
    train_indices, val_indices = train_test_split(range(len(train_set)), test_size=0.2, stratify=np.array([train_set[i][1] for i in range(len(train_set))]), random_state=cfg.config.seed) 

    # Crea i sottoinsiemi per training e validazione
    f_train_set = Subset(train_set, train_indices)
    val_set = Subset(train_set, val_indices)
    
    # Crea i DataLoader per il caricamento efficiente dei dati in batch
    # shuffle=True per il training: mescola i dati ad ogni epoca per evitare overfitting
    # generator con seed garantisce riproducibilità completa dello shuffle
    # drop_last=True evita errori con BatchNorm quando l'ultimo batch ha solo 1 elemento
    train_loader = DataLoader(f_train_set, batch_size=cfg.config.batch_size, shuffle=True, generator=generator, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.config.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=cfg.config.batch_size, shuffle=False, drop_last=False)

    # Crea l'oggetto NetRunner che gestisce training e testing
    netrunner = NetRunner(cfg)
    
    # Esegue training oppure solo testing in base alla configurazione
    if cfg.config.training:
        print('=' * 50)
        print('INIZIO TRAINING...')
        print('=' * 50)
        netrunner.train(train_loader, val_loader)
        print('=' * 50)
        print('TRAINING COMPLETATO - INIZIO TESTING...')
        print('=' * 50)
        netrunner.test(test_loader, use_current_model=True)
    else:
        print('Testing con il modello salvato...')
        netrunner.test(test_loader)




if __name__ == '__main__':
    data_file, schema_file = Path('./config.json'), Path('./config_schema.json')
    valid_input = data_file.is_file() and schema_file.is_file() and data_file.suffix == '.json' and schema_file.suffix == '.json'
 
    if valid_input: # Validazione del config.json      
        with open(Path(data_file)) as d:
            with open(Path(schema_file)) as s:
                data, schema = json.load(d), json.load(s)
                
                try:
                    jsonschema.validate(instance=data, schema=schema)                    
                except jsonschema.exceptions.ValidationError:
                    print(f'Json config file is not following schema rules.')
                    sys.exit(-1)
                except jsonschema.exceptions.SchemaError:
                    print(f'Json config schema file is invalid.')
                    sys.exit(-1)

   
    with open(Path(data_file)) as d:
        main(json.loads(d.read(), object_hook=lambda d: SimpleNamespace(**d)))
    
    