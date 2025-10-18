import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Optional, Tuple

from .plot_utils import PlotColorManager


class CostPlotter:
    """Plota métricas de custo computacional.

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

    def plot_model_parameter_counts(
        self,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> None:
        """Plota número de parâmetros por modelo (menor = melhor)."""
        if models is None:
            models = sorted(self.results.keys())

        param_counts = []
        for m in models:
            info = self.results[m].additional_info
            if not info or 'n_parameters' not in info:
                raise ValueError(f"O modelo '{m}' não possui 'n_parameters' em additional_info.")
            param_counts.append(info['n_parameters'])

        # índice do menor número de parâmetros
        best_idx = int(np.argmin(param_counts))
        x = np.arange(len(models))
        width = 0.6

        fig, ax = plt.subplots(figsize=figsize)
        for i, (cnt, m) in enumerate(zip(param_counts, models)):
            color = self._get_color(m)
            bar = ax.bar(x[i], cnt, width, color=color, alpha=0.9)
            ax.text(bar[0].get_x()+bar[0].get_width()/2, cnt*1.01,
                    f"{int(cnt):,}", ha='center', va='bottom', fontsize=9)
            if i == best_idx:
                bar[0].set_edgecolor('k')
                bar[0].set_linewidth(2)
                bar[0].set_linestyle('--')

        legend_patches = [Patch(color=self._get_color(m), label=m, alpha=0.9) for m in models]
        ax.set_title('Quantidade de Parâmetros por Modelo', fontsize=14)
        ax.set_ylabel('Número de Parâmetros', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        max_cnt = max(param_counts)
        ax.set_ylim(0, max_cnt * 1.15)
        ax.grid(alpha=0.3, axis='y')
        ax.legend(handles=legend_patches, title='Modelos',
                  bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_inference_cost(
        self,
        metric: str = "latency_ms",  # 'latency_ms' | 'throughput_ps' | 'peak_mem_mb'
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Gráfico de barras para custo computacional por modelo.
        Barras de erro = quantis p5–p95 (assimétricas).
        Procura primeiro em compute_cost['summary'][key] e, se faltar,
        calcula a partir de compute_cost['per_run'].

        Para memória, prefere 'peak_mem_reserved_mb' (mais estável).
        """
        if models is None:
            models = sorted(self.results.keys())

        # Mapeia métrica -> chaves esperadas
        if metric == "latency_ms":
            title = "Custo Computacional – Latência"
            ylab = "ms por patch"
            fmt = lambda x: f"{x:.1f}"
            key_summary = key_perrun = key_legacy = "latency_ms_per_patch"
            lower_is_better = True
        elif metric == "throughput_ps":
            title = "Custo Computacional – Vazão"
            ylab = "patches / s"
            fmt = lambda x: f"{x:.1f}"
            key_summary = key_perrun = key_legacy = "throughput_patches_per_s"
            lower_is_better = False
        elif metric == "peak_mem_mb":
            title = "Custo Computacional – Memória Pico (GPU)"
            ylab = "MB"
            fmt = lambda x: f"{x:.0f}"
            # Preferimos 'reserved' (summary/per_run); recuamos p/ legados
            key_summary = "peak_mem_reserved_mb"
            key_perrun = "peak_mem_reserved_mb"
            key_legacy = "peak_mem_mb"  # legado (alocado)
            lower_is_better = True
        else:
            raise ValueError("metric deve ser 'latency_ms', 'throughput_ps' ou 'peak_mem_mb'.")

        centers, err_low, err_high, used_models = [], [], [], []

        for m in models:
            res = self.results.get(m)
            info = getattr(res, "additional_info", {}) or {}
            cc = info.get("compute_cost", {})
            if not cc:
                continue

            # 1) Tenta summary: median, p5, p95
            summ = cc.get("summary", {}) or {}
            block = summ.get(key_summary, {})
            med = block.get("median", None)
            p5 = block.get("p5", None)
            p95 = block.get("p95", None)

            # 2) Fallback: valor legado único (sem dispersão)
            if med is None:
                v = cc.get(key_legacy, None) if metric == "peak_mem_mb" else cc.get(key_summary, None)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    med = float(v)

            # 3) Se faltar p5/p95 (ou até a mediana), calcula de per_run
            if (p5 is None or p95 is None) or (med is None):
                runs = cc.get("per_run", []) or []
                vals = []
                for r in runs:
                    if metric == "peak_mem_mb":
                        v = r.get("peak_mem_reserved_mb", None)
                        if v is None:
                            v = r.get("peak_mem_alloc_mb", None)  # outro nome possível
                        if v is None:
                            v = r.get("peak_mem_mb", None)        # último recurso
                    else:
                        v = r.get(key_perrun, None)
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        vals.append(float(v))

                if len(vals) >= 1:
                    qs = np.percentile(vals, [5, 50, 95])
                    if med is None:
                        med = float(qs[1])
                    if p5 is None:
                        p5 = float(qs[0])
                    if p95 is None:
                        p95 = float(qs[2])

            # 4) Consolida
            if med is None or (isinstance(med, float) and math.isnan(med)):
                continue

            # Erros assimétricos: med - p5  /  p95 - med
            e_low = float(med - p5) if (p5 is not None) else 0.0
            e_high = float(p95 - med) if (p95 is not None) else 0.0
            e_low = max(0.0, e_low)
            e_high = max(0.0, e_high)

            centers.append(float(med))
            err_low.append(e_low)
            err_high.append(e_high)
            used_models.append(m)

        if not used_models:
            raise ValueError("Nenhum modelo possui dados de 'compute_cost'. Execute a avaliação com benchmark.")

        # Índice do "melhor" (lat/mem: menor; throughput: maior)
        best_idx = int(np.argmin(centers)) if lower_is_better else int(np.argmax(centers))

        # ---- Plot
        x = np.arange(len(used_models))
        width = 0.6
        fig, ax = plt.subplots(figsize=figsize)

        # Limites brutos via p5–p95 (antes de anotar) e margem p/ colocar o texto acima do erro
        data_ymin = min(v - lo for v, lo in zip(centers, err_low))
        data_ymax = max(v + hi for v, hi in zip(centers, err_high))
        rng = (data_ymax - data_ymin) if data_ymax != data_ymin else (abs(data_ymax) or 1.0)
        pad = 0.02 * rng  # margem para o rótulo acima da barra de erro

        bars = []
        for i, (m, v, lo, hi) in enumerate(zip(used_models, centers, err_low, err_high)):
            color = self._get_color(m)
            bar = ax.bar(x[i], v, width, color=color, alpha=0.9)
            bars.append(bar)

            # Barras de erro p5–p95 (assimétricas), se existirem
            if (lo > 0) or (hi > 0):
                ax.errorbar(x[i], v,
                            yerr=np.array([[lo], [hi]]),
                            fmt='none', capsize=4, linewidth=1.5,
                            color='k', zorder=3)

            # destaque da melhor barra
            if i == best_idx:
                bar[0].set_edgecolor('k')
                bar[0].set_linewidth(2)
                bar[0].set_linestyle('--')

        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylab, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(used_models, rotation=30, ha='right')

        # Limites do eixo Y considerando p5–p95 e espaço para o rótulo
        ymin = data_ymin
        ymax = data_ymax
        headroom = 0.15 * (ymax - ymin) + pad  # 15% + margem do rótulo
        if ymax > 0:
            ax.set_ylim(0 if ymin >= 0 else ymin - headroom, ymax + headroom)
        else:
            ax.set_ylim(ymin - headroom, ymax + headroom)

        # Rótulo da mediana *acima* do topo da barra de erro (v + hi)
        for i, (v, hi) in enumerate(zip(centers, err_high)):
            y_text = v + hi + pad
            ax.text(x[i], y_text, fmt(v), ha='center', va='bottom', fontsize=9, zorder=5)

        legend_patches = [Patch(color=self._get_color(m), label=m, alpha=0.9) for m in used_models]
        ax.grid(alpha=0.3, axis='y')
        ax.legend(handles=legend_patches, title='Modelos',
                  bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
