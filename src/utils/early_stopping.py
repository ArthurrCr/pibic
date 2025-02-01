class EarlyStopping:
    """
    Implementação simples de Early Stopping:
    - Monitora uma métrica (loss ou IoU, por exemplo).
    - Para se 'mode' for 'min', busca o menor valor da métrica (ex: loss).
    - Para se 'mode' for 'max', busca o maior valor da métrica (ex: IoU).
    - Se não houver melhora por 'patience' épocas consecutivas, interrompe o treinamento.
    """
    def __init__(self, patience=2, mode='min', min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_metric = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_metric):
        if self.best_metric is None:
            # primeira epoch
            self.best_metric = current_metric
            return
        
        if self.mode == 'min':
            improvement = (self.best_metric - current_metric) > self.min_delta
            if improvement:
                self.best_metric = current_metric
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
            improvement = (current_metric - self.best_metric) > self.min_delta
            if improvement:
                self.best_metric = current_metric
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
