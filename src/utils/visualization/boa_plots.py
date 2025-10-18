import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Optional, Tuple

from .plot_utils import PlotColorManager, EXPERIMENTS


class BOAPlotter:
    """Plota métricas BOA e otimização de limiares.

    Compatível com o comportamento do results_manager.py:
    - Usa o mapa de cores do manager (se existir _get_color), ou
      cai no PlotColorManager local para manter estabilidade.
    """

    def __init__(self, results_manager, color_manager: Optional[PlotColorManager] = None):
        # Guarda referência e acesso aos resultados
        self._rm = results_manager
        self.results = results_manager.results

        # Compatibilização de cores com o monolítico:
        # - se o manager tiver _get_color (como no results_manager.py), use-o;
        # - caso contrário, use (ou crie) um PlotColorManager local.
        if hasattr(results_manager, "_get_color") and callable(getattr(results_manager, "_get_color")):
            self._get_color = results_manager._get_color
            self.color_manager = None
        else:
            self.color_manager = color_manager or PlotColorManager()
            self._get_color = self.color_manager.get_color

    def plot_boa(
        self,
        experiment: str,
        use_optimal: bool = False,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compara BOA entre modelos (baseline ou t*) e destaca a maior barra.
        Lógica e saída compatíveis com results_manager.py.
        """
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Experimento '{experiment}' não reconhecido.")

        if models is None:
            models = sorted(self.results.keys())

        boas, labels, colors = [], [], []
        for mdl in models:
            res = self.results[mdl]

            if use_optimal:
                if experiment not in res.optimal_thresholds:
                    raise ValueError(f"{mdl} não possui t* para '{experiment}'.")
                boa = res.optimal_thresholds[experiment]['best_median_boa']
                thr = res.optimal_thresholds[experiment]['best_threshold']
                label = f"{mdl} (t*={thr:.2f})"
            else:
                if experiment not in res.boa_baseline:
                    raise ValueError(f"{mdl} não possui BOA‑baseline para '{experiment}'.")
                boa = res.boa_baseline[experiment]
                label = mdl

            boas.append(boa)
            labels.append(label)
            colors.append(self._get_color(mdl))

        # Índice do melhor (maior BOA)
        best_idx = int(np.argmax(boas))

        # Plot
        x, width = np.arange(len(models)), 0.6
        fig, ax = plt.subplots(figsize=figsize)

        for i, (boa, lbl, col) in enumerate(zip(boas, labels, colors)):
            bar = ax.bar(x[i], boa, width, color=col, alpha=0.9)  # sem label
            ax.text(bar[0].get_x()+bar[0].get_width()/2, boa+0.003,
                    f"{boa:.4f}", ha='center', va='bottom', fontsize=9)
            if i == best_idx:  # Melhor barra
                bar[0].set_edgecolor('k')
                bar[0].set_linewidth(2)
                bar[0].set_linestyle('--')

        # Legenda (sem tracejado nas amostras)
        legend_patches = [Patch(color=col, label=lbl, alpha=0.9)
                          for col, lbl in zip(colors, labels)]

        tipo = 'BOA (t*)' if use_optimal else 'BOA (argmax)'
        ax.set_title(f'{tipo} – Experimento: {experiment}', fontsize=14)
        ax.set_ylabel('BOA', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.set_ylim(0, max(boas) * 1.15)
        ax.grid(alpha=0.3, axis='y')
        ax.legend(handles=legend_patches, title='Modelos',
                  bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_optimal_threshold_curve(
        self,
        model_name: str,
        experiment: str,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> None:
        """Plota curva de mediana BOA vs. limiar para um experimento (idêntico ao monolítico)."""
        if experiment not in self.results[model_name].optimal_thresholds:
            print(f"Experimento '{experiment}' não encontrado para {model_name}")
            return

        data = self.results[model_name].optimal_thresholds[experiment]

        plt.figure(figsize=figsize)
        plt.plot(data['thresholds'], data['median_boas'], linewidth=2)
        plt.scatter(data['best_threshold'], data['best_median_boa'],
                    s=100, zorder=5,
                    label=f"t* = {data['best_threshold']:.2f}")

        plt.xlabel('Limiar', fontsize=12)
        plt.ylabel('Mediana BOA', fontsize=12)
        plt.title(f'Curva de Otimização de Limiar – {experiment} – {model_name}',
                  fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)

        info = (f"BOA máximo: {data['best_median_boa']:.4f}\n"
                f"N patches: {data['n_patches']}")
        plt.text(0.05, 0.95, info, transform=plt.gca().transAxes,
                 va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_boa_comparison_table(
        self,
        model_name: str,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> None:
        """Tabela resumindo BOA baseline versus BOA com t* (mesma formatação do monolítico)."""
        data = {
            'Experimento': EXPERIMENTS,
            'BOA (argmax)': [],
            'Limiar Ótimo (t*)': [],
            'BOA (t*)': [],
            'Melhoria': []
        }

        res = self.results[model_name]
        for exp in EXPERIMENTS:
            boa_base = res.boa_baseline.get(exp, np.nan)
            data['BOA (argmax)'].append(f"{boa_base:.4f}" if not np.isnan(boa_base) else '-')

            if exp in res.optimal_thresholds:
                opt = res.optimal_thresholds[exp]
                data['Limiar Ótimo (t*)'].append(f"{opt['best_threshold']:.2f}")
                data['BOA (t*)'].append(f"{opt['best_median_boa']:.4f}")
                diff = opt['best_median_boa'] - boa_base
                data['Melhoria'].append(f"{diff:+.4f}")
            else:
                data['Limiar Ótimo (t*)'].append('-')
                data['BOA (t*)'].append('-')
                data['Melhoria'].append('-')

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        # Cores linha-a-linha na coluna "Melhoria"
        cell_colours = []
        for m in data['Melhoria']:
            row = ['white'] * len(df.columns)
            if m not in ('-', '+0.0000', '-0.0000'):
                row[-1] = '#90EE90' if m[0] == '+' else '#FFB6C1'
            cell_colours.append(row)

        table = ax.table(cellText=df.values, colLabels=df.columns,
                         cellColours=cell_colours, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.25, 2)

        for c in range(len(df.columns)):
            table[(0, c)].set_facecolor('#3498db')
            table[(0, c)].set_text_props(color='white', weight='bold')

        plt.title(f'Comparação BOA – {model_name}', pad=20, fontsize=14, weight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
