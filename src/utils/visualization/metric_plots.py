import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Optional, Tuple

from .plot_utils import PlotColorManager, CLASS_NAMES


class MetricPlotter:
    """Plota métricas de avaliação por classe.

    Compatível com o comportamento do results_manager.py:
    - Reutiliza o mapa de cores do manager (se existir _get_color)
      para manter consistência visual com o monolítico; cai em
      PlotColorManager local se não houver.
    """

    def __init__(self, results_manager, color_manager: Optional[PlotColorManager] = None):
        # Referências
        self._rm = results_manager
        self.results = results_manager.results

        # Compatibilização de cores com o monolítico
        if hasattr(results_manager, "_get_color") and callable(getattr(results_manager, "_get_color")):
            self._get_color = results_manager._get_color
            self.color_manager = None
        else:
            self.color_manager = color_manager or PlotColorManager()
            self._get_color = self.color_manager.get_color

    def plot_individual_metric(
        self,
        metric: str,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 12),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plota determinada métrica por classe comparando modelos.
        Destaca (contorno preto) a barra com melhor valor em cada classe.
        """
        if models is None:
            models = sorted(self.results.keys())

        # Verificação de dados (mesma validação do monolítico)
        for m in models:
            for cname in CLASS_NAMES:
                if (cname not in self.results[m].metrics or
                    metric not in self.results[m].metrics[cname]):
                    raise ValueError(
                        f"Métrica '{metric}' ausente para classe '{cname}' no modelo '{m}'."
                    )

        # Matriz valores[modelo, classe]
        values_mat = np.array([
            [self.results[m].metrics[c][metric] for c in CLASS_NAMES]
            for m in models
        ])

        # Índice do "melhor" em cada classe
        if metric in ("Omission Error", "Commission Error"):
            best_idx = values_mat.argmin(axis=0)   # menor = melhor
        else:
            best_idx = values_mat.argmax(axis=0)   # maior = melhor

        # Plot
        n_models = len(models)
        width = 0.8 / n_models
        x = np.arange(len(CLASS_NAMES))

        fig, ax = plt.subplots(figsize=figsize)

        for i, model in enumerate(models):
            vals = values_mat[i]
            offset = (i - n_models / 2 + 0.5) * width
            color = self._get_color(model)

            bars = ax.bar(x + offset, vals, width, alpha=0.85, color=color)

            # Labels e contorno
            for j, bar in enumerate(bars):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=9)
                if i == best_idx[j]:  # melhor dessa classe
                    bar.set_edgecolor('k')
                    bar.set_linewidth(2)
                    bar.set_linestyle('--')

        # Legenda (patches sem tracejado)
        legend_patches = [Patch(color=self._get_color(m), label=m, alpha=0.85)
                          for m in models]

        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Comparação de {metric} entre Modelos', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.legend(handles=legend_patches, title='Modelos',
                  bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_errors_for_class(
        self,
        class_name: str,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plota EO (omission) e EC (commission) de uma única classe
        para diversos modelos.
        """
        if class_name not in CLASS_NAMES:
            raise ValueError(f"Classe '{class_name}' não reconhecida.")

        if models is None:
            models = sorted(self.results.keys())

        # Verifica se todos os modelos têm os campos necessários
        for m in models:
            if class_name not in self.results[m].metrics:
                raise ValueError(f"{m} não contém métricas para '{class_name}'.")
            for mt in ('Omission Error', 'Commission Error'):
                if mt not in self.results[m].metrics[class_name]:
                    raise ValueError(
                        f"Métrica '{mt}' ausente no modelo '{m}' "
                        f"para a classe '{class_name}'."
                    )

        # Dados
        eo_vals = [self.results[m].metrics[class_name]['Omission Error'] for m in models]
        ec_vals = [self.results[m].metrics[class_name]['Commission Error'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - width/2, eo_vals, width,
                       label='Erro de Omissão', color='#ff7f0e', alpha=0.9)
        bars2 = ax.bar(x + width/2, ec_vals, width,
                       label='Erro de Comissão', color='#1f77b4', alpha=0.9)

        # Rótulos acima das barras
        for bars in (bars1, bars2):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=9)

        # Estética
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.set_ylabel('Erro', fontsize=12)
        ax.set_title(f'Erros de Omissão vs. Comissão – {class_name}', fontsize=14)
        ax.legend()
        ax.set_ylim(0, max(eo_vals + ec_vals) * 1.15)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
