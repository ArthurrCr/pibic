import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from typing import List, Optional, Tuple

from .plot_utils import format_model_name, CLASS_NAMES


class QualitativePlotter:
    """Plota exemplos qualitativos e análises."""

    def __init__(self, results_manager):
        # Guarda o manager para acessar selection_info e results
        self._rm = results_manager
        self.results = results_manager.results

        # Cores para as classes (seguem CLASS_NAMES)
        self.class_colors = ['#2E7D32', '#B71C1C', '#F57C00', '#6A1B9A']
        self.cmap = ListedColormap(self.class_colors)
        self.norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

        # Mapeia nomes de categorias para o índice de classe
        # (aceita "Cloud Shadow" e "Shadow" como a mesma classe 3)
        self.category_to_class = {
            'Clear': 0,
            'Thick Cloud': 1,
            'Thin Cloud': 2,
            'Shadow': 3,
            'Cloud Shadow': 3,
            'Mixed/Transition': None
        }

    def plot_qualitative_examples(
        self,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 16),
        save_path: Optional[str] = None,
        categories_to_show: Optional[List[str]] = None
    ) -> None:
        """
        Visualização robusta com métricas IoU/OE/CE por classe‑alvo.

        Parameters
        ----------
        models : list, optional
            Lista de modelos a visualizar; se None, usa todos.
        figsize : tuple
            Tamanho da figura (largura, altura).
        save_path : str, optional
            Caminho para salvar a figura.
        categories_to_show : list, optional
            Lista de categorias para filtrar e mostrar.
            Ex.: ['Clear'], ['Thin Cloud', 'Shadow'].
        """
        if models is None:
            models = sorted(self.results.keys())

        # Filtra apenas modelos que possuem exemplos qualitativos
        models_with_examples = []
        for model in models:
            if (model in self.results and
                hasattr(self.results[model], 'additional_info') and
                'qualitative_examples' in self.results[model].additional_info):
                examples = self.results[model].additional_info['qualitative_examples']
                # Aceita lista ou dicionário; precisa ter algum conteúdo
                has_data = False
                if isinstance(examples, list):
                    has_data = len(examples) > 0
                elif isinstance(examples, dict):
                    has_data = any(bool(examples.get(k)) for k in ('typical_errors',
                                                                   'ambiguous_cases',
                                                                   'edge_cases'))
                if has_data:
                    models_with_examples.append(model)

        if not models_with_examples:
            print("Nenhum modelo possui exemplos qualitativos coletados.")
            return

        n_models = len(models_with_examples)

        # Pega os exemplos do primeiro modelo como referência
        first = self.results[models_with_examples[0]].additional_info['qualitative_examples']

        # Normaliza: se for dicionário (typical/ambiguous/edge), achata para lista
        if isinstance(first, dict):
            pooled = []
            for key in ('typical_errors', 'ambiguous_cases', 'edge_cases'):
                if key in first and first[key]:
                    pooled.extend(first[key])
            first_model_examples = pooled
        else:
            first_model_examples = first

        if not first_model_examples:
            print("Nenhum exemplo qualitativo disponível.")
            return

        # Filtra por categorias, se solicitado
        if categories_to_show:
            filtered = []
            for ex in first_model_examples:
                cat = ex.get('category', '')
                if any(c in cat for c in categories_to_show):
                    filtered.append(ex)
            if not filtered:
                print(f"Nenhum exemplo encontrado para as categorias: {categories_to_show}")
                return
            examples_to_show = filtered
        else:
            # Por padrão, mostra até 5 exemplos
            examples_to_show = first_model_examples[:5]

        n_examples = len(examples_to_show)

        # Layout: horizontal se uma única categoria, senão vertical
        if categories_to_show and len(categories_to_show) == 1:
            self._plot_horizontal_layout(
                examples_to_show, models_with_examples, n_examples,
                n_models, figsize, save_path, categories_to_show
            )
        else:
            self._plot_vertical_layout(
                examples_to_show, models_with_examples, n_examples,
                n_models, figsize, save_path, categories_to_show
            )

        # Análise textual para o paper
        self._print_qualitative_analysis_for_paper(
            models_with_examples, n_examples, categories_to_show
        )

    # ------------------------- Layouts ------------------------- #

    def _plot_horizontal_layout(
        self, examples_to_show, models_with_examples, n_examples,
        n_models, figsize, save_path, categories_to_show
    ):
        """Layout horizontal (linhas = exemplos; colunas = RGB/GT + cada modelo)."""
        n_rows = n_examples
        n_cols = 3 + n_models  # Categoria, RGB, GT, + predições dos modelos

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for ex_idx, example in enumerate(examples_to_show):
            # Converte CHW -> HWC, prioriza bandas 4-3-2 se houver 4+ canais
            img_data = np.asarray(example['image'])
            if img_data.ndim == 3:
                if img_data.shape[0] >= 4:
                    rgb = img_data[[3, 2, 1], :, :].transpose(1, 2, 0)
                elif img_data.shape[0] == 3:
                    rgb = img_data.transpose(1, 2, 0)
                else:
                    rgb = np.repeat(img_data, 3, axis=0).transpose(1, 2, 0)
            else:
                continue

            # Normalização e leve “boost” para visualização
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            rgb = np.clip(rgb * 1.5, 0, 1)

            true_mask = np.squeeze(example['true_mask'])
            category = example.get('category', f'Exemplo {ex_idx+1}')

            # Classe‑alvo (para mostrar IoU/OE/CE quando aplicável)
            target_class = None
            for cat_name, class_id in self.category_to_class.items():
                if cat_name in category:
                    target_class = class_id
                    break

            # Coluna 0: Categoria/Label
            ax = axes[ex_idx, 0]
            ax.text(0.5, 0.5, category, ha='center', va='center',
                    fontsize=11, fontweight='bold', color='darkblue',
                    transform=ax.transAxes)
            ax.axis('off')
            if ex_idx == 0:
                ax.set_title('Categoria', fontsize=10, fontweight='bold')

            # Coluna 1: RGB
            ax = axes[ex_idx, 1]
            ax.imshow(rgb)
            ax.axis('off')
            if ex_idx == 0:
                ax.set_title('RGB', fontsize=10, fontweight='bold')

            # Coluna 2: Ground Truth
            ax = axes[ex_idx, 2]
            ax.imshow(true_mask, cmap=self.cmap, norm=self.norm, interpolation='nearest')
            ax.axis('off')
            if ex_idx == 0:
                ax.set_title('Ground Truth', fontsize=10, fontweight='bold')

            # Colunas 3+: Predições dos modelos
            for model_idx, model_name in enumerate(models_with_examples):
                model_examples = self.results[model_name].additional_info['qualitative_examples']

                # Procura o exemplo correspondente por (batch_idx, image_idx)
                corresponding_example = None
                # Aceita lista ou dicionário
                if isinstance(model_examples, dict):
                    iterable = []
                    for key in ('typical_errors', 'ambiguous_cases', 'edge_cases'):
                        if key in model_examples and model_examples[key]:
                            iterable.extend(model_examples[key])
                else:
                    iterable = model_examples

                for m_ex in iterable:
                    if (m_ex['batch_idx'] == example['batch_idx'] and
                        m_ex['image_idx'] == example['image_idx']):
                        corresponding_example = m_ex
                        break

                ax = axes[ex_idx, 3 + model_idx]

                if corresponding_example:
                    pred_mask = np.squeeze(corresponding_example['pred_mask'])
                    class_metrics = corresponding_example.get('class_metrics', {})

                    # Predição
                    ax.imshow(pred_mask, cmap=self.cmap, norm=self.norm, interpolation='nearest')

                    # Métricas (IoU/OE/CE) ou Accuracy (Mixed)
                    if target_class is not None and target_class in class_metrics:
                        m = class_metrics[target_class]
                        iou, oe, ce = m['iou'], m['oe'], m['ce']
                        color = 'green' if iou > 0.7 else 'orange' if iou > 0.5 else 'red'
                        metrics_text = f'IoU:{iou:.3f}\nOE:{oe:.2f} CE:{ce:.2f}'
                        ax.text(0.5, -0.08, metrics_text,
                                transform=ax.transAxes, ha='center', va='top',
                                fontsize=8, fontweight='bold', color=color,
                                linespacing=1.5)
                    elif target_class is None:  # Mixed/Transition
                        acc = corresponding_example.get('accuracy', None)
                        if acc is not None:
                            color = 'green' if acc > 0.9 else 'orange' if acc > 0.7 else 'red'
                            ax.text(0.5, -0.05, f'Acc: {acc:.3f}',
                                    transform=ax.transAxes, ha='center', va='top',
                                    fontsize=9, fontweight='bold', color=color)

                ax.axis('off')
                if ex_idx == 0:
                    formatted_name = format_model_name(model_name)
                    fontsize = 8 if len(model_name) > 25 else 9
                    ax.set_title(formatted_name, fontsize=fontsize, fontweight='bold',
                                 multialignment='center')

        # Legenda das classes
        legend_elements = [mpatches.Patch(color=self.class_colors[i], label=CLASS_NAMES[i])
                           for i in range(4)]
        fig.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True,
                   fontsize=11, title='Classes', title_fontsize=12)

        # Título geral
        title = f'Análise Qualitativa: {categories_to_show[0]}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.05)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_vertical_layout(
        self, examples_to_show, models_with_examples, n_examples,
        n_models, figsize, save_path, categories_to_show
    ):
        """Layout vertical (colunas = exemplos; linhas = RGB/GT + cada modelo)."""
        n_rows = 2 + n_models  # RGB, GT + modelos
        n_cols = n_examples

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for ex_idx, example in enumerate(examples_to_show):
            # Converte CHW -> HWC
            img_data = np.asarray(example['image'])
            if img_data.ndim == 3:
                if img_data.shape[0] >= 4:
                    rgb = img_data[[3, 2, 1], :, :].transpose(1, 2, 0)
                elif img_data.shape[0] == 3:
                    rgb = img_data.transpose(1, 2, 0)
                else:
                    rgb = np.repeat(img_data, 3, axis=0).transpose(1, 2, 0)
            else:
                continue

            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            rgb = np.clip(rgb * 1.5, 0, 1)

            true_mask = np.squeeze(example['true_mask'])
            category = example.get('category', f'Exemplo {ex_idx+1}')

            # Classe‑alvo
            target_class = None
            for cat_name, class_id in self.category_to_class.items():
                if cat_name in category:
                    target_class = class_id
                    break

            # Linha 0: RGB
            ax = axes[0, ex_idx]
            ax.imshow(rgb)
            ax.set_title(f'{category}', fontsize=9, fontweight='bold', color='darkblue')
            ax.axis('off')
            if ex_idx == 0:
                ax.set_ylabel('RGB', fontsize=10, fontweight='bold')
                ax.yaxis.set_label_coords(-0.15, 0.5)

            # Linha 1: Ground Truth
            ax = axes[1, ex_idx]
            ax.imshow(true_mask, cmap=self.cmap, norm=self.norm, interpolation='nearest')
            ax.axis('off')
            if ex_idx == 0:
                ax.set_ylabel('Ground Truth', fontsize=10, fontweight='bold')
                ax.yaxis.set_label_coords(-0.15, 0.5)

            # Linhas 2+: Predições de cada modelo
            for model_idx, model_name in enumerate(models_with_examples):
                model_examples = self.results[model_name].additional_info['qualitative_examples']
                # Aceita lista ou dicionário
                if isinstance(model_examples, dict):
                    iterable = []
                    for key in ('typical_errors', 'ambiguous_cases', 'edge_cases'):
                        if key in model_examples and model_examples[key]:
                            iterable.extend(model_examples[key])
                else:
                    iterable = model_examples

                corresponding_example = None
                for m_ex in iterable:
                    if (m_ex['batch_idx'] == example['batch_idx'] and
                        m_ex['image_idx'] == example['image_idx']):
                        corresponding_example = m_ex
                        break

                if corresponding_example:
                    pred_mask = np.squeeze(corresponding_example['pred_mask'])
                    class_metrics = corresponding_example.get('class_metrics', {})

                    ax = axes[2 + model_idx, ex_idx]
                    ax.imshow(pred_mask, cmap=self.cmap, norm=self.norm, interpolation='nearest')

                    # Métricas
                    if target_class is not None and target_class in class_metrics:
                        m = class_metrics[target_class]
                        metrics_text = f"IoU:{m['iou']:.2f}\nOE:{m['oe']:.2f} CE:{m['ce']:.2f}"
                        color = 'green' if m['iou'] > 0.7 else 'orange' if m['iou'] > 0.5 else 'red'
                    else:
                        acc = corresponding_example.get('accuracy', None)
                        metrics_text = f'Acc:{acc:.0%}' if acc is not None else 'Acc: N/A'
                        color = 'green' if (acc is not None and acc > 0.9) else \
                                'orange' if (acc is not None and acc > 0.7) else 'red'

                    ax.text(0.5, -0.15, metrics_text,
                            transform=ax.transAxes, ha='center', va='top',
                            fontsize=8, fontweight='bold', color=color,
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white', alpha=0.8, edgecolor=color))

                    ax.axis('off')
                    if ex_idx == 0:
                        formatted_name = format_model_name(model_name)
                        fontsize = 8 if len(model_name) > 25 else 9
                        ax.set_ylabel(formatted_name, fontsize=fontsize, fontweight='bold',
                                      multialignment='center')
                        ax.yaxis.set_label_coords(-0.20 if '\n' in formatted_name else -0.15, 0.5)

        # Legenda das classes
        legend_elements = [mpatches.Patch(color=self.class_colors[i], label=CLASS_NAMES[i])
                           for i in range(4)]
        fig.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True,
                   fontsize=11, title='Classes', title_fontsize=12)

        # Título
        if categories_to_show and len(categories_to_show) == 1:
            title = f'Análise Qualitativa: {categories_to_show[0]}'
        else:
            title = 'Análise Qualitativa: Seleção por Quantis com Métricas IoU/OE/CE'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.05)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # --------------------- Saída textual p/ paper --------------------- #

    def _print_qualitative_analysis_for_paper(
        self, models: List[str], n_examples: int, categories_to_show: Optional[List[str]] = None
    ) -> None:
        """
        Imprime análise detalhada dos exemplos qualitativos para inclusão no paper.
        """
        print("\n" + "="*60)
        print("ANÁLISE PARA O PAPER (Tabela/Texto)")
        print("="*60)

        # Categorias disponíveis (nome canônico)
        all_categories = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Shadow', 'Mixed/Transition']

        # Normaliza categorias solicitadas (aceita 'Cloud Shadow' como 'Shadow')
        synonym = {'Cloud Shadow': 'Shadow'}
        if categories_to_show is None:
            categories = all_categories
        else:
            categories = []
            for cat in categories_to_show:
                cat_norm = synonym.get(cat, cat)
                if cat_norm in all_categories and cat_norm not in categories:
                    categories.append(cat_norm)
            if not categories:
                print("Nenhuma categoria válida especificada.")
                return

        # Tabela comparativa por categoria (IoU ou Acc no caso Mixed)
        print(f"\nTabela 1: Métricas por categoria e modelo")
        print("-" * 80)
        header = f"{'Modelo':<30}" + "".join(f"{(c[:12] + ('…' if len(c) > 12 else '')):<15}" for c in categories)
        print(header)
        print("-" * 80)

        best_per_category = {cat: (0.0, None) for cat in categories}

        for model_name in models:
            if 'qualitative_examples' not in self.results[model_name].additional_info:
                continue

            # Usa até n_examples para cada modelo
            model_examples_full = self.results[model_name].additional_info['qualitative_examples']
            # Aceita lista ou dicionário
            if isinstance(model_examples_full, dict):
                model_examples = []
                for key in ('typical_errors', 'ambiguous_cases', 'edge_cases'):
                    if key in model_examples_full and model_examples_full[key]:
                        model_examples.extend(model_examples_full[key])
            else:
                model_examples = model_examples_full
            model_examples = model_examples[:n_examples]

            display_name = model_name[:28] + '..' if len(model_name) > 30 else model_name
            row = f"{display_name:<30}"

            for cat in categories:
                # Procura o primeiro exemplo dessa categoria
                cat_examples = [ex for ex in model_examples if cat in ex.get('category', '') or
                                (cat == 'Shadow' and 'Cloud Shadow' in ex.get('category', ''))]

                if cat_examples:
                    ex = cat_examples[0]
                    target_class = self.category_to_class[cat]

                    if target_class is not None and 'class_metrics' in ex and target_class in ex['class_metrics']:
                        iou = ex['class_metrics'][target_class]['iou']
                        row += f"{iou:.3f}          "
                        if iou > best_per_category[cat][0]:
                            best_per_category[cat] = (iou, model_name)
                    else:
                        # Mixed/Transition → accuracy
                        acc = ex.get('accuracy', 0.0)
                        row += f"{acc:.3f}          "
                        if acc > best_per_category[cat][0]:
                            best_per_category[cat] = (acc, model_name)
                else:
                    row += f"-              "

            print(row)

        print("-" * 80)

        # Vencedores por categoria
        print("\nMelhor desempenho por categoria:")
        for cat in categories:
            if best_per_category[cat][1]:
                metric_name = "IoU" if self.category_to_class[cat] is not None else "Acc"
                print(f"  {cat}: {best_per_category[cat][1]} ({metric_name}={best_per_category[cat][0]:.3f})")

        # Análise OE×CE para categorias críticas
        critical = ['Thin Cloud', 'Shadow']
        critical_to_analyze = [c for c in critical if c in categories]
        if critical_to_analyze:
            print("\n" + "=" * 60)
            print("ANÁLISE OE×CE NOS CASOS CRÍTICOS")
            print("=" * 60)
            for model_name in models:
                if 'qualitative_examples' not in self.results[model_name].additional_info:
                    continue

                # Até n_examples por modelo
                model_examples_full = self.results[model_name].additional_info['qualitative_examples']
                if isinstance(model_examples_full, dict):
                    model_examples = []
                    for key in ('typical_errors', 'ambiguous_cases', 'edge_cases'):
                        if key in model_examples_full and model_examples_full[key]:
                            model_examples.extend(model_examples_full[key])
                else:
                    model_examples = model_examples_full
                model_examples = model_examples[:n_examples]

                print(f"\n{model_name}:")

                for target_cat in critical_to_analyze:
                    target_class = self.category_to_class[target_cat]
                    cat_examples = [ex for ex in model_examples if target_cat in ex.get('category', '') or
                                    (target_cat == 'Shadow' and 'Cloud Shadow' in ex.get('category', ''))]
                    if cat_examples and target_class is not None:
                        ex = cat_examples[0]
                        if 'class_metrics' in ex and target_class in ex['class_metrics']:
                            m = ex['class_metrics'][target_class]
                            print(f"  {target_cat}:")
                            print(f"    IoU: {m['iou']:.3f}")
                            print(f"    OE (FN rate): {m['oe']:.3f} {'← Low is good' if m['oe'] < 0.3 else '← High!'}")
                            print(f"    CE (FP rate): {m['ce']:.3f} {'← Low is good' if m['ce'] < 0.3 else '← High!'}")

        # Protocolo de seleção (se o manager possuir esta informação)
        if hasattr(self._rm, 'selection_info'):
            info = getattr(self._rm, 'selection_info', None)
            if isinstance(info, dict):
                print("\n" + "=" * 60)
                print("PROTOCOLO DE SELEÇÃO (para mencionar no paper):")
                print("=" * 60)
                print("1. Análise completa do conjunto de teste (sem seed - conjunto fixo)")
                print("2. Limiares baseados em quantis:")
                t = info.get('thresholds', {})
                if t:
                    print(f"   - Thin Cloud: ≥{t.get('thin_cloud', 0):.2f} (Q75)")
                    print(f"   - Shadow: ≥{t.get('shadow', 0):.2f} (Q75)")
                    print(f"   - Thick Cloud: ≥{t.get('thick_cloud', 0):.2f} (Q50)")
                    print(f"   - Clear: ≥{t.get('clear', 0):.2f} (Q80)")
                    print(f"   - Mixed (boundary density): ≥{t.get('mixed_boundary', 0):.3f} (Q80)")
                print("3. Seleção: mediano do top-10 por score de dificuldade (entropia+discordância normalizadas)")
                print("4. Métricas exibidas: IoU, OE (omission), CE (commission) da classe-alvo")
