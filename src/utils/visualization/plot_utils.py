import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


class PlotColorManager:
    """Gerencia cores consistentes para modelos.

    Mantém um mapeamento determinístico model_name -> cor usando a paleta Set3
    (12 cores distintas). Quando a quantidade de modelos ultrapassa 12,
    as cores são reutilizadas ciclicamente mantendo estabilidade de sessão.
    """

    def __init__(self):
        self._color_map: Dict[str, tuple] = {}
        # Paleta (Set3 tem 12 cores distintas; expande-se ciclicamente se precisar)
        self._palette = plt.cm.Set3(np.linspace(0, 1, 12))

    def get_color(self, model_name: str) -> tuple:
        """Retorna sempre a mesma cor para um modelo (cria se necessário)."""
        if model_name not in self._color_map:
            idx = len(self._color_map) % len(self._palette)
            self._color_map[model_name] = self._palette[idx]
        return self._color_map[model_name]

    def reset(self) -> None:
        """Limpa o mapa de cores."""
        self._color_map.clear()


# =========================
# Constantes globais
# =========================
# Mantidas idênticas às do results_manager.py para compatibilidade
CLASS_NAMES = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Cloud Shadow']
METRICS_NAMES = ['F1-Score', 'Precision', 'Recall', 'Omission Error', 'Commission Error']
EXPERIMENTS = ['cloud/no cloud', 'cloud shadow', 'valid/invalid']

# Cores (hex) para classes de segmentação, seguindo a mesma ordem de CLASS_NAMES
CLASS_COLORS = {
    0: '#2E7D32',  # Clear - verde
    1: '#B71C1C',  # Thick Cloud - vermelho
    2: '#F57C00',  # Thin Cloud - laranja
    3: '#6A1B9A'   # Cloud Shadow - roxo
}


def format_model_name(name: str) -> str:
    """Adiciona quebras de linha inteligentes no nome do modelo para títulos/legendas.

    Regras:
    - Quebra em '(PCA' quando presente (ex.: '(PCA / 7)').
    - Se muito longo, quebra no primeiro '('.
    - Se ainda muito longo, quebra no primeiro '+'.
    """
    # Quebrar em "(PCA" - funciona para (PCA / 7), (PCA / 5), etc.
    if '(PCA' in name:
        parts = name.split('(PCA')
        return parts[0].strip() + '\n(PCA' + parts[1]
    # Quebrar em "(" genérico se nome muito longo
    elif '(' in name and len(name) > 25:
        parts = name.split('(', 1)
        return parts[0].strip() + '\n(' + parts[1]
    # Quebrar em "+" se muito longo
    elif '+' in name and len(name) > 30:
        parts = name.split('+', 1)
        return parts[0].strip() + '\n+ ' + parts[1].strip()
    return name
