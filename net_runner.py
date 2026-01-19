"""
Classe NetRunner: gestisce il training e il testing della ResNet.
Contiene gli ottimizzatori, la loss function e la logica di training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from net import ResNet

# Imposta semi fissi per la riproducibilità dei risultati
torch.manual_seed(42)
np.random.seed(42)

class NetRunner:
    """
    Classe responsabile di training, testing e gestione del modello ResNet.
    """
    
    def __init__(self, cfg):
        """
        Inizializza la rete, l'ottimizzatore e gli altri componenti.
        Args:
            cfg: oggetto di configurazione contenente iperparametri
        """
        # Usa GPU se disponibile, altrimenti CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Crea la rete ResNet e la sposta sul dispositivo (GPU o CPU)
        self.model = ResNet(num_classes=10).to(self.device)
        
        # Ottimizzatore Adam: aggiorna i pesi della rete durante il training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.config.learning_rate)
        
        # Funzione di perdita per la classificazione multiclasse
        self.criterion = nn.CrossEntropyLoss()
        
        # Scheduler: riduce il learning rate periodicamente per migliorare la convergenza
        if cfg.config.scheduler.isActive:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=cfg.config.scheduler.step_size, 
                gamma=cfg.config.scheduler.gamma
            )
        
        self.cfg = cfg
        

    def train(self, train_loader, val_loader):
        """
        Funzione principale di training.
        Allena la rete per num_epochs iterazioni.
        """
        num_epochs = self.cfg.config.num_epochs
        es_start_epoch = self.cfg.config.early_stopping.start_epoch
        es_loss_evolution_epochs = self.cfg.config.early_stopping.loss_evolution_epochs
        es_patience = self.cfg.config.early_stopping.patience
        es_improvement_rate = self.cfg.config.early_stopping.improvement_rate
        
        # Variabili per tracciare le perdite migliori
        best_tr_loss = float('inf')
        best_va_loss = float('inf')
        
        early_stop_check = False
        
        # Lista per salvare la perdita media di ogni epoca
        epoch_loss_values = []
        va_loss_no_improve = 0

        # Loop su tutte le epoche
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            # Attiva l'early stopping a partire da una determinata epoca
            if (epoch + 1) == es_start_epoch and self.cfg.config.early_stopping.isActive:
                early_stop_check = True
            
            # Loop su tutti i batch dell'epoca
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Modalità training: abilita dropout e batch normalization
                self.model.train()
                
                # Sposta i dati sul dispositivo (GPU o CPU)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass: calcola le predizioni
                outputs = self.model(inputs)
                
                # Calcola la perdita (error)
                loss = self.criterion(outputs, targets)
                
                # Backward pass: calcola i gradienti
                self.optimizer.zero_grad()  # Azzera i gradienti precedenti
                loss.backward()              # Calcola i nuovi gradienti
                self.optimizer.step()        # Aggiorna i pesi
                
                # Accumula la perdita per il calcolo della media
                running_loss += loss.item()
                
                # Stampa la perdita ad ogni batch
                print(f'[Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}] Loss: {loss.item():.4f}')

            # Applica lo scheduler per ridurre il learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Calcola la perdita media dell'epoca
            avg_epoch_loss = running_loss / len(train_loader)
            epoch_loss_values.append(avg_epoch_loss)
            
            # Se la perdita è migliorata, salva il modello
            if avg_epoch_loss < best_tr_loss:
                best_tr_loss = avg_epoch_loss
                print('Save best model...')
                torch.save(self.model.state_dict(), './out/best_model_sd.pth')
                torch.save(self.model, './out/best_model.pth')
            
            # Early stopping: valida periodicamente
            if early_stop_check and (epoch + 1) % es_loss_evolution_epochs == 0:
                print('Validating...')
                val_loss = self.test(val_loader, use_current_model=True, validation=True)
                
                # Controlla il miglioramento della validazione
                if val_loss < best_va_loss:
                    # Calcola il tasso di miglioramento percentuale
                    improve_ratio = abs((val_loss / best_va_loss) - 1) * 100
                    
                    # Se il miglioramento è significativo, aggiorna la miglior loss
                    if improve_ratio >= es_improvement_rate:
                        best_va_loss = val_loss
                        va_loss_no_improve = 0
                    else:
                        va_loss_no_improve += 1
                else:
                    # Nessun miglioramento rilevato
                    va_loss_no_improve += 1
            
            # Early stopping: se nessun miglioramento per 'patience' epoche, ferma il training
            if va_loss_no_improve >= es_patience:
                print(f"Early stopping all'epoca {epoch + 1}")
                break

        # Salva il modello addestrato finale
        torch.save(self.model.state_dict(), './out/model_final_sd.pth')  # Solo i parametri
        torch.save(self.model, './out/model_final.pth')                  # Intero modello
        print('Modello salvato')

        # Crea un grafico della perdita per ogni epoca
        print("Generando grafico della perdita...")
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, 
                 marker='o', linestyle='-', color='r', label='Average Epoch Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Epoch')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def getModel(self):
        """
        Permette all'utente di scegliere quale modello caricare dalla cartella other_models.
        """
        p = Path(self.cfg.io.choice_model_path)
        if p.exists() and p.is_dir():
            files = [file.name for file in p.iterdir() if file.is_file()]
            
            print("File disponibili:")
            for idx, file in enumerate(files, 1):
                print(f"{idx}. {file}")
            
            # Loop fino a che l'utente non sceglie un file valido
            while True:
                try:
                    scelta = int(input("Scegli il numero del file desiderato: ")) - 1
                    if 0 <= scelta < len(files):
                        file_scelto = files[scelta]
                        print(f"Hai scelto: {file_scelto}")
                        return str(file_scelto)
                    else:
                        print(f"Errore: inserisci un numero tra 1 e {len(files)}.")
                except ValueError:
                    print("Errore: inserisci un numero valido.")
            
        else:
            print(f"La cartella non esiste o non è una directory.")
        

    def test(self, dataloader, use_current_model=False, validation=False):
        """
        Testa la rete su un dataset (test o validazione).
        Args:
            dataloader: DataLoader con i dati di test/validazione
            use_current_model: se True, usa il modello in memoria; se False, carica da file
            validation: se True, testa su validazione; se False, su test
        """
        # Carica il modello appropriato
        if use_current_model:
            model = self.model
        else:
            # Se non usiamo il modello attuale, creane uno nuovo e caricalo da file
            model = ResNet(num_classes=10).to(self.device)
            path = self.cfg.io.choice_model_path + '/' + self.getModel()
            model.load_state_dict(torch.load(path))
        
        # Modalità valutazione: disabilita dropout e batch normalization
        model.eval()

        test_loss = 0
        correct = 0
        total = 0

        # Non calcolare i gradienti durante il test (risparmia memoria)
        with torch.no_grad():
            for inputs, targets in dataloader:
                # Sposta i dati sul dispositivo
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = model(inputs)
                
                # Calcola la perdita
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                # Prende la classe con la probabilità più alta
                _, predicted = outputs.max(1)
                
                # Conta i risultati corretti
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Calcola l'accuratezza media
        accuracy = 100. * correct / total
        average_loss = test_loss / len(dataloader)
        
        # Stampa i risultati
        if validation:
            print(f'Validation set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
            return average_loss
        else:
            print(f'Test set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')

        
