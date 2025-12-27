import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, field
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt

# Constantes globais
CLASS_NAMES   = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Cloud Shadow']
METRICS_NAMES = ['F1-Score', 'Precision', 'Recall',
                 'Omission Error', 'Commission Error']
EXPERIMENTS   = ['cloud/no cloud', 'cloud shadow', 'valid/invalid']


@dataclass
class ModelResult:
    """Estrutura para armazenar resultados de um modelo."""
    metrics: Dict
    confusion_matrix: np.ndarray
    overall_accuracy: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    boa_baseline: Dict = field(default_factory=dict)
    optimal_thresholds: Dict = field(default_factory=dict)
    additional_info: Dict = field(default_factory=dict)


class ModelResultsManager:
    """Gerencia resultados de modelos e cria visualizações comparativas."""

    def __init__(self):
        self.results: Dict[str, ModelResult] = {}

       
        self._color_map: Dict[str, tuple] = {}
        # Paleta (Set3 tem 12 cores distintas; expande‑se ciclicamente se precisar)
        self._palette = plt.cm.Set3(np.linspace(0, 1, 12))
 

    def _get_color(self, model_name: str):
        """Retorna sempre a mesma cor para um modelo (cria se necessário)."""
       
        if model_name not in self._color_map:
            idx = len(self._color_map) % len(self._palette)
            self._color_map[model_name] = self._palette[idx]
        return self._color_map[model_name]
       

    def save_model_results(self, model_name: str, metrics: Dict,
                           conf_matrix: np.ndarray, overall_accuracy: float,
                           additional_info: Dict | None = None):
        """Salva resultados de um modelo usando dataclass."""
        self.results[model_name] = ModelResult(
            metrics=metrics,
            confusion_matrix=conf_matrix,
            overall_accuracy=overall_accuracy,
            additional_info=additional_info or {}
        )

    def parse_metrics_from_output(self, model_name: str,
                                  metrics_dict: Dict,
                                  conf_matrix: np.ndarray):
        """Converte dicionário da avaliação em estrutura própria."""
        parsed = {}
        for c in CLASS_NAMES:
            if c in metrics_dict:
                parsed[c] = {k: metrics_dict[c][k]
                             for k in METRICS_NAMES + ['Support']
                             if k in metrics_dict[c]}
        self.save_model_results(model_name, parsed, conf_matrix,
                                metrics_dict['Overall']['Accuracy'])

    def save_boa_results(self, model_name: str,
                         df_results: pd.DataFrame | None = None,
                         threshold_results: Dict | None = None,
                         experiment: str | None = None):
        """Salva BOA baseline e/ou resultados de limiar ótimo."""
        if model_name not in self.results:
            self.results[model_name] = ModelResult(
                metrics={}, confusion_matrix=np.array([]), overall_accuracy=0.0
            )

        # Baseline a partir do DataFrame
        if df_results is not None:
            for _, row in df_results.iterrows():
                exp = row['Experiment']
                self.results[model_name].boa_baseline[exp] = float(row['Median BOA'])

        # Resultado do t*
        if threshold_results is not None and experiment is not None:
            self.results[model_name].optimal_thresholds[experiment] = threshold_results


    def plot_individual_metric(
            self, metric: str,
            models: List[str] | None = None,
            figsize: tuple = (20, 12),
            save_path: str | None = None,
    ):
        """
        Plota determinada métrica por classe comparando modelos.
        Destaca (contorno preto) a barra com melhor valor em cada classe.
        """
        from matplotlib.patches import Patch
        
        if models is None:
            models = sorted(self.results.keys())

        #–– verificação de dados
        for m in models:
            for cname in CLASS_NAMES:
                if (cname not in self.results[m].metrics or
                    metric not in self.results[m].metrics[cname]):
                    raise ValueError(
                        f"Métrica '{metric}' ausente para classe '{cname}' no modelo '{m}'."
                    )

        #–– matriz valores[modelo, classe]
        values_mat = np.array([
            [self.results[m].metrics[c][metric] for c in CLASS_NAMES]
            for m in models
        ])

        #–– índice do "melhor" em cada classe
        if metric in ("Omission Error", "Commission Error"):
            best_idx = values_mat.argmin(axis=0)   # menor é melhor
        else:
            best_idx = values_mat.argmax(axis=0)   # maior é melhor

        #–– plot
        n_models = len(models)
        width    = 0.8 / n_models
        x        = np.arange(len(CLASS_NAMES))

        fig, ax = plt.subplots(figsize=figsize)

        for i, model in enumerate(models):
            vals   = values_mat[i]
            offset = (i - n_models / 2 + 0.5) * width
            color  = self._get_color(model)

            bars = ax.bar(x + offset, vals, width, alpha=0.85, color=color)  # sem label

            # labels e contorno
            for j, bar in enumerate(bars):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=9)
                if i == best_idx[j]:                  # é a melhor dessa classe
                    bar.set_edgecolor('k')
                    bar.set_linewidth(2)
                    bar.set_linestyle('--')  

        # Cria patches para a legenda (sem tracejado)
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


    def plot_optimal_threshold_curve(self, model_name: str, experiment: str,
                                     figsize: tuple = (10, 6),
                                     save_path: str | None = None):
        """Plota curva de mediana BOA vs. limiar para um experimento."""
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

    def plot_boa_comparison_table(self, model_name: str,
                                  figsize: tuple = (10, 6),
                                  save_path: str | None = None):
        """Tabela resumindo BOA baseline versus BOA com t*."""
        data = {'Experimento': EXPERIMENTS,
                'BOA (argmax)': [], 'Limiar Ótimo (t*)': [],
                'BOA (t*)': [], 'Melhoria': []}

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

        # cores linha‑a‑linha na coluna "Melhoria"
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

    def generate_summary_report(self, models: List[str] | None = None) -> pd.DataFrame:
        """Gera DataFrame‑resumo (acc, BOA médio etc.) para modelos selecionados."""
        if models is None:
            models = sorted(self.results.keys())

        rows = []
        for mdl in models:
            r = self.results[mdl]
            row = {'Modelo': mdl,
                   'Acurácia Global': f"{r.overall_accuracy:.4f}",
                   'Timestamp': r.timestamp[:10]}

            if r.boa_baseline:
                row['BOA Baseline Médio'] = (f"{np.mean(list(r.boa_baseline.values())):.4f}")

            if r.optimal_thresholds:
                row['BOA Otimizado Médio'] = (
                    f"{np.mean([v['best_median_boa'] for v in r.optimal_thresholds.values()]):.4f}")

            rows.append(row)

        return pd.DataFrame(rows)

    def plot_errors_for_class(
        self,
        class_name: str,
        models: List[str] | None = None,
        figsize: tuple = (12, 6),
        save_path: str | None = None,
        ):
        """
        Plota EO (omission) e EC (commission) de uma única classe
        para diversos modelos.

        Parameters
        ----------
        class_name : str
            Nome exato da classe (tem que existir em CLASS_NAMES).
        models : list[str], optional
            Lista de modelos a comparar.  Se None, usa todos.
        figsize : tuple
            Tamanho da figura.
        save_path : str, optional
            Caminho para salvar o PNG se desejado.
        """
        if class_name not in CLASS_NAMES:
            raise ValueError(f"Classe '{class_name}' não reconhecida.")

        if models is None:
            models = sorted(self.results.keys())

        # Verifica se todos os modelos têm os campos necessários
        for m in models:
            if class_name not in self.results[m].metrics:
                raise ValueError(f"{m} não contém métricas para '{class_name}'.")
            for metric in ('Omission Error', 'Commission Error'):
                if metric not in self.results[m].metrics[class_name]:
                    raise ValueError(
                        f"Métrica '{metric}' ausente no modelo '{m}' "
                        f"para a classe '{class_name}'."
                    )

        # Dados
        eo_vals = [self.results[m].metrics[class_name]['Omission Error']  for m in models]
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

    def plot_boa(
            self,
            experiment: str,
            use_optimal: bool = False,
            models: List[str] | None = None,
            figsize: tuple = (10, 6),
            save_path: str | None = None,
    ):
        """
        Compara BOA entre modelos (baseline ou t*) e destaca a maior barra.
        """
        from matplotlib.patches import Patch
        
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

        #–– índice do melhor (maior BOA)
        best_idx = int(np.argmax(boas))

        #–– plot
        x, width = np.arange(len(models)), 0.6
        fig, ax  = plt.subplots(figsize=figsize)
        for i, (boa, lbl, col) in enumerate(zip(boas, labels, colors)):
            bar = ax.bar(x[i], boa, width, color=col, alpha=0.9)  # sem label
            ax.text(bar[0].get_x()+bar[0].get_width()/2, boa+0.003,
                    f"{boa:.4f}", ha='center', va='bottom', fontsize=9)
            if i == best_idx:                      # melhor barra
                bar[0].set_edgecolor('k')
                bar[0].set_linewidth(2)
                bar[0].set_linestyle('--')  

        # Cria patches para a legenda (sem tracejado)
        legend_patches = [Patch(color=col, label=lbl, alpha=0.9) 
                        for col, lbl in zip(colors, labels)]

        tipo = 'BOA (t*)' if use_optimal else 'BOA (argmax)'
        ax.set_title(f'{tipo} – Experimento: {experiment}', fontsize=14)
        ax.set_ylabel('BOA', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.set_ylim(0, max(boas)*1.15)
        ax.grid(alpha=0.3, axis='y')
        ax.legend(handles=legend_patches, title='Modelos', 
                bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    

    def plot_model_parameter_counts(self, models=None, figsize=(10, 6), save_path=None):
        if models is None:
            models = sorted(self.results.keys())

        param_counts = []
        for m in models:
            info = self.results[m].additional_info
            if not info or 'n_parameters' not in info:
                raise ValueError(f"O modelo '{m}' não possui 'n_parameters' em additional_info.")
            param_counts.append(info['n_parameters'])

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

    def _print_qualitative_analysis_for_paper(self, models, n_examples, categories_to_show=None):
        """
        Imprime análise detalhada dos exemplos qualitativos para inclusão no paper
        
        Parameters
        ----------
        models : list
            Lista de modelos para analisar
        n_examples : int
            Número de exemplos para analisar
        categories_to_show : list, optional
            Lista de categorias para mostrar na tabela. 
            Se None, mostra todas as 5 categorias.
            Ex: ['Thin Cloud', 'Shadow', 'Mixed/Transition']
        """
        print("\n" + "="*60)
        print("ANÁLISE PARA O PAPER (Tabela/Texto)")
        print("="*60)
        
        # Categorias disponíveis
        all_categories = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Shadow', 'Mixed/Transition']
        
        # Usar categorias especificadas ou todas
        if categories_to_show is None:
            categories = all_categories
        else:
            # Validar categorias
            categories = [cat for cat in categories_to_show if cat in all_categories]
            if not categories:
                print("Nenhuma categoria válida especificada.")
                return
        
        category_to_class = {
            'Clear': 0,
            'Thick Cloud': 1,
            'Thin Cloud': 2,
            'Shadow': 3,
            'Mixed/Transition': None  # Usa accuracy ao invés de IoU
        }
        
        # Tabela comparativa de IoU por categoria
        print(f"\nTabela 1: Métricas por categoria e modelo")
        print("-"*80)
        
        # Header
        header = f"{'Modelo':<30}"
        for cat in categories:
            if len(cat) > 12:
                header += f"{cat[:12]:<15}"
            else:
                header += f"{cat:<15}"
        print(header)
        print("-"*80)
        
        # Dados por modelo
        best_per_category = {cat: (0, None) for cat in categories}
        
        for model_name in models:
            if 'qualitative_examples' not in self.results[model_name].additional_info:
                continue
                
            model_examples = self.results[model_name].additional_info['qualitative_examples'][:n_examples]
            
            # Truncar nome do modelo se necessário
            display_name = model_name[:28] + '..' if len(model_name) > 30 else model_name
            row = f"{display_name:<30}"
            
            for cat in categories:
                # Encontrar exemplo dessa categoria
                cat_examples = [ex for ex in model_examples if cat in ex.get('category', '')]
                
                if cat_examples:
                    ex = cat_examples[0]
                    target_class = category_to_class[cat]
                    
                    if target_class is not None and 'class_metrics' in ex:
                        if target_class in ex['class_metrics']:
                            iou = ex['class_metrics'][target_class]['iou']
                            row += f"{iou:.3f}          "
                            
                            # Rastrear melhor modelo
                            if iou > best_per_category[cat][0]:
                                best_per_category[cat] = (iou, model_name)
                        else:
                            row += f"N/A            "
                    else:
                        # Para Mixed, usar accuracy
                        row += f"{ex['accuracy']:.3f}          "
                        if ex['accuracy'] > best_per_category[cat][0]:
                            best_per_category[cat] = (ex['accuracy'], model_name)
                else:
                    row += f"-              "
            
            print(row)
        
        print("-"*80)
        
        # Destacar vencedores
        print("\nMelhor desempenho por categoria:")
        for cat in categories:
            if best_per_category[cat][1]:
                metric_name = "IoU" if category_to_class[cat] is not None else "Acc"
                print(f"  {cat}: {best_per_category[cat][1]} ({metric_name}={best_per_category[cat][0]:.3f})")
        
        # Análise de OE×CE apenas para categorias críticas
        critical_categories = ['Thin Cloud', 'Shadow']
        critical_to_analyze = [cat for cat in critical_categories if cat in categories]
        
        if critical_to_analyze:
            print("\n" + "="*60)
            print("ANÁLISE OE×CE NOS CASOS CRÍTICOS")
            print("="*60)
            
            for model_name in models:
                if 'qualitative_examples' not in self.results[model_name].additional_info:
                    continue
                    
                model_examples = self.results[model_name].additional_info['qualitative_examples'][:n_examples]
                
                print(f"\n{model_name}:")
                
                for target_cat in critical_to_analyze:
                    target_class = category_to_class[target_cat]
                    cat_examples = [ex for ex in model_examples if target_cat in ex.get('category', '')]
                    
                    if cat_examples:
                        ex = cat_examples[0]
                        if 'class_metrics' in ex and target_class in ex['class_metrics']:
                            m = ex['class_metrics'][target_class]
                            print(f"  {target_cat}:")
                            print(f"    IoU: {m['iou']:.3f}")
                            print(f"    OE (FN rate): {m['oe']:.3f} {'← Low is good' if m['oe'] < 0.3 else '← High!'}")
                            print(f"    CE (FP rate): {m['ce']:.3f} {'← Low is good' if m['ce'] < 0.3 else '← High!'}")
        
        # Informações sobre o protocolo de seleção se disponível
        if hasattr(self, 'selection_info'):
            print("\n" + "="*60)
            print("PROTOCOLO DE SELEÇÃO (para mencionar no paper):")
            print("="*60)
            print("1. Análise completa do conjunto de teste (sem seed - conjunto fixo)")
            print("2. Limiares baseados em quantis:")
            
            info = self.selection_info
            if 'thresholds' in info:
                t = info['thresholds']
                print(f"   - Thin Cloud: ≥{t.get('thin_cloud', 0):.2f} (Q75)")
                print(f"   - Shadow: ≥{t.get('shadow', 0):.2f} (Q75)")
                print(f"   - Thick Cloud: ≥{t.get('thick_cloud', 0):.2f} (Q50)")
                print(f"   - Clear: ≥{t.get('clear', 0):.2f} (Q80)")
                print(f"   - Mixed (boundary density): ≥{t.get('mixed_boundary', 0):.3f} (Q80)")
            
            print("3. Seleção: mediano do top-10 por score de dificuldade (entropia+discordância normalizadas)")
            print("4. Métricas exibidas: IoU, OE (omission), CE (commission) da classe-alvo")

    def plot_qualitative_examples(self, models=None, figsize=(20, 16), save_path=None, 
                                categories_to_show=None):
        """
        Visualização robusta com métricas IoU/OE/CE por classe-alvo
        
        Parameters
        ----------
        models : list, optional
            Lista de modelos para visualizar
        figsize : tuple
            Tamanho da figura
        save_path : str, optional
            Caminho para salvar a figura
        categories_to_show : list, optional
            Lista de categorias para filtrar E mostrar.
            Ex: ['Clear'] mostra apenas Clear em layout horizontal
            Ex: ['Thin Cloud', 'Shadow'] mostra apenas essas duas categorias
            Se None, mostra todas as 5 categorias disponíveis.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap, BoundaryNorm
        import numpy as np
        
        def format_model_name(name):
            """Adiciona quebras de linha inteligentes no nome do modelo"""
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
        
        if models is None:
            models = sorted(self.results.keys())
        
        # Verificar modelos com exemplos
        models_with_examples = []
        for model in models:
            if (model in self.results and 
                hasattr(self.results[model], 'additional_info') and
                'qualitative_examples' in self.results[model].additional_info):
                examples = self.results[model].additional_info['qualitative_examples']
                if examples and len(examples) > 0:
                    models_with_examples.append(model)
        
        if not models_with_examples:
            print("Nenhum modelo possui exemplos qualitativos coletados.")
            return
        
        # Cores para as classes 
        colors = ['#2E7D32', '#B71C1C', '#F57C00', '#6A1B9A']  # Clear(verde), Thick(vermelho), Thin(laranja), Shadow(roxo)
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)
        
        # Mapear categoria para classe
        category_to_class = {
            'Clear': 0,
            'Thick Cloud': 1,
            'Thin Cloud': 2,
            'Shadow': 3,
            'Mixed/Transition': None
        }
        
        n_models = len(models_with_examples)
        
        # Pegar todos os exemplos do primeiro modelo
        first_model_examples = self.results[models_with_examples[0]].additional_info['qualitative_examples']
        
        # Filtrar exemplos por categorias se especificado
        if categories_to_show:
            filtered_examples = []
            for ex in first_model_examples:
                cat = ex.get('category', '')
                # Verificar se a categoria está na lista desejada
                if any(c in cat for c in categories_to_show):
                    filtered_examples.append(ex)
            
            if not filtered_examples:
                print(f"Nenhum exemplo encontrado para as categorias: {categories_to_show}")
                return
            
            examples_to_show = filtered_examples
        else:
            examples_to_show = first_model_examples[:5]  # Máximo 5 exemplos
        
        n_examples = len(examples_to_show)
        
        # Decidir layout baseado no número de categorias
        if categories_to_show and len(categories_to_show) == 1:
            # Layout horizontal para categoria única
            n_rows = n_examples
            n_cols = 3 + n_models  # RGB, GT, Mask + modelos
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for ex_idx, example in enumerate(examples_to_show):
                # Processar imagem para RGB
                img_data = np.asarray(example['image'])
                if img_data.ndim == 3:  # CHW
                    if img_data.shape[0] >= 4:
                        # Sentinel-2: B4,B3,B2 para RGB
                        rgb = img_data[[3,2,1], :, :].transpose(1,2,0)
                    elif img_data.shape[0] == 3:
                        rgb = img_data.transpose(1,2,0)
                    else:
                        rgb = np.repeat(img_data, 3, axis=0).transpose(1,2,0)
                else:
                    continue
                
                # Normalizar para visualização
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                rgb = np.clip(rgb * 1.5, 0, 1)
                
                true_mask = np.squeeze(example['true_mask'])
                category = example.get('category', f'Exemplo {ex_idx+1}')
                
                # Identificar classe alvo
                target_class = None
                for cat_name, class_id in category_to_class.items():
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
                ax.imshow(true_mask, cmap=cmap, norm=norm, interpolation='nearest')
                ax.axis('off')
                if ex_idx == 0:
                    ax.set_title('Ground Truth', fontsize=10, fontweight='bold')
                
                # Colunas 3+: Predições de cada modelo
                for model_idx, model_name in enumerate(models_with_examples):
                    model_examples = self.results[model_name].additional_info['qualitative_examples']
                    
                    # Encontrar o exemplo correspondente
                    corresponding_example = None
                    for m_ex in model_examples:
                        if (m_ex['batch_idx'] == example['batch_idx'] and 
                            m_ex['image_idx'] == example['image_idx']):
                            corresponding_example = m_ex
                            break
                    
                    ax = axes[ex_idx, 3 + model_idx]
                    
                    if corresponding_example:
                        pred_mask = np.squeeze(corresponding_example['pred_mask'])
                        class_metrics = corresponding_example.get('class_metrics', {})
                        
                        # Mostrar predição
                        ax.imshow(pred_mask, cmap=cmap, norm=norm, interpolation='nearest')
                        
                        # Calcular e mostrar IoU, OE e CE (com quebra de linha)
                        if target_class is not None and target_class in class_metrics:
                            m = class_metrics[target_class]
                            iou = m['iou']
                            oe = m['oe']
                            ce = m['ce']
                            
                            # Determinar cor baseada no IoU
                            color = 'green' if iou > 0.7 else 'orange' if iou > 0.5 else 'red'
                            
                            # Texto com IoU em uma linha e OE/CE na linha abaixo
                            metrics_text = f'IoU:{iou:.3f}\nOE:{oe:.2f} CE:{ce:.2f}'
                            
                            ax.text(0.5, -0.08, metrics_text, 
                                transform=ax.transAxes, ha='center', va='top',
                                fontsize=8, fontweight='bold', color=color,
                                linespacing=1.5)
                                
                        elif target_class is None:  # Mixed/Transition
                            acc = corresponding_example['accuracy']
                            color = 'green' if acc > 0.9 else 'orange' if acc > 0.7 else 'red'
                            ax.text(0.5, -0.05, f'Acc: {acc:.3f}', 
                                transform=ax.transAxes, ha='center', va='top',
                                fontsize=9, fontweight='bold', color=color)
                    
                    ax.axis('off')
                    if ex_idx == 0:
                        # Nome do modelo com quebras de linha inteligentes
                        formatted_name = format_model_name(model_name)
                        fontsize = 8 if len(model_name) > 25 else 9
                        ax.set_title(formatted_name, fontsize=fontsize, fontweight='bold',
                                multialignment='center')
            
        else:
            # Layout tradicional (vertical) para múltiplas categorias
            n_rows = 2 + n_models  # RGB, GT + modelos
            n_cols = n_examples
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for ex_idx, example in enumerate(examples_to_show):
                # Processar imagem para RGB
                img_data = np.asarray(example['image'])
                if img_data.ndim == 3:  # CHW
                    if img_data.shape[0] >= 4:
                        rgb = img_data[[3,2,1], :, :].transpose(1,2,0)
                    elif img_data.shape[0] == 3:
                        rgb = img_data.transpose(1,2,0)
                    else:
                        rgb = np.repeat(img_data, 3, axis=0).transpose(1,2,0)
                else:
                    continue
                
                # Normalizar para visualização
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                rgb = np.clip(rgb * 1.5, 0, 1)
                
                true_mask = np.squeeze(example['true_mask'])
                category = example.get('category', f'Exemplo {ex_idx+1}')
                
                # Identificar classe alvo
                target_class = None
                for cat_name, class_id in category_to_class.items():
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
                ax.imshow(true_mask, cmap=cmap, norm=norm, interpolation='nearest')
                ax.axis('off')
                if ex_idx == 0:
                    ax.set_ylabel('Ground Truth', fontsize=10, fontweight='bold')
                    ax.yaxis.set_label_coords(-0.15, 0.5)
                
                # Linhas 2+: Predições de cada modelo
                for model_idx, model_name in enumerate(models_with_examples):
                    model_examples = self.results[model_name].additional_info['qualitative_examples']
                    
                    # Encontrar o exemplo correspondente
                    corresponding_example = None
                    for m_ex in model_examples:
                        if (m_ex['batch_idx'] == example['batch_idx'] and 
                            m_ex['image_idx'] == example['image_idx']):
                            corresponding_example = m_ex
                            break
                    
                    if corresponding_example:
                        pred_mask = np.squeeze(corresponding_example['pred_mask'])
                        class_metrics = corresponding_example.get('class_metrics', {})
                        
                        ax = axes[2 + model_idx, ex_idx]
                        ax.imshow(pred_mask, cmap=cmap, norm=norm, interpolation='nearest')
                        
                        # Métricas (já estava com quebra de linha)
                        if target_class is not None and target_class in class_metrics:
                            m = class_metrics[target_class]
                            metrics_text = f"IoU:{m['iou']:.2f}\n"
                            metrics_text += f"OE:{m['oe']:.2f} CE:{m['ce']:.2f}"
                            color = 'green' if m['iou'] > 0.7 else 'orange' if m['iou'] > 0.5 else 'red'
                        else:
                            acc = corresponding_example['accuracy']
                            metrics_text = f'Acc:{acc:.0%}'
                            color = 'green' if acc > 0.9 else 'orange' if acc > 0.7 else 'red'
                        
                        ax.text(0.5, -0.15, metrics_text, 
                            transform=ax.transAxes, ha='center', va='top',
                            fontsize=8, fontweight='bold', color=color,
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', alpha=0.8, edgecolor=color))
                        
                        ax.axis('off')
                        if ex_idx == 0:
                            # Nome do modelo com quebras de linha
                            formatted_name = format_model_name(model_name)
                            fontsize = 8 if len(model_name) > 25 else 9
                            ax.set_ylabel(formatted_name, fontsize=fontsize, fontweight='bold',
                                        multialignment='center')
                            ax.yaxis.set_label_coords(-0.20 if '\n' in formatted_name else -0.15, 0.5)
        
        # Adicionar legenda das classes
        legend_elements = [mpatches.Patch(color=colors[i], label=CLASS_NAMES[i]) for i in range(4)]
        fig.legend(handles=legend_elements, loc='upper center', 
                bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True, 
                fontsize=11, title='Classes', title_fontsize=12)
        
        # Título geral
        if categories_to_show and len(categories_to_show) == 1:
            title = f'Análise Qualitativa: {categories_to_show[0]}'
        else:
            title = 'Análise Qualitativa: Seleção por Quantis com Métricas IoU/OE/CE'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        
        # Ajustar espaçamento
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.05)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Análise quantitativa para o paper
        self._print_qualitative_analysis_for_paper(models_with_examples, n_examples, 
                                                categories_to_show=categories_to_show)


    def plot_aggregated_metrics_table(self, models=None, figsize=(14, 8), save_path=None):
        """
        Plota tabela comparativa das métricas agregadas por categoria
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if models is None:
            models = sorted(self.results.keys())
        
        # Verificar quais modelos têm métricas agregadas
        models_with_metrics = []
        for model in models:
            if (model in self.results and 
                hasattr(self.results[model], 'additional_info') and
                'aggregated_metrics_by_category' in self.results[model].additional_info):
                models_with_metrics.append(model)
        
        if not models_with_metrics:
            print("Nenhum modelo possui métricas agregadas por categoria.")
            print("Execute evaluate_model() primeiro.")
            return
        
        # Preparar dados para a tabela
        categories = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Shadow', 'Mixed/Trans.']
        table_data = []
        
        # Rastrear melhor valor por categoria
        best_per_category = {cat: 0.0 for cat in categories}
        
        for model_name in models_with_metrics:
            metrics = self.results[model_name].additional_info['aggregated_metrics_by_category']
            
            # Truncar nome se necessário
            display_name = model_name[:30] + '..' if len(model_name) > 32 else model_name
            
            row = [display_name]
            for cat in categories:
                # Ajustar nome da categoria para busca
                full_cat = cat if cat != 'Mixed/Trans.' else 'Mixed/Transition'
                
                if full_cat in metrics and metrics[full_cat]['count'] > 0:
                    mean_val = metrics[full_cat]['mean']
                    count = metrics[full_cat]['count']
                    row.append(f"{mean_val:.3f}\n({count})")
                    
                    # Rastrear melhor
                    if mean_val > best_per_category[cat]:
                        best_per_category[cat] = mean_val
                else:
                    row.append("---\n(0)")
            
            table_data.append(row)
        
        # Criar figura e tabela
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        # Preparar cores das células
        cell_colors = []
        for row_idx, row in enumerate(table_data):
            row_colors = ['#f0f0f0']  # Cor para nome do modelo
            
            for col_idx, cat in enumerate(categories):
                cell_value = row[col_idx + 1]
                
                # Extrair valor numérico
                if '---' not in cell_value:
                    value = float(cell_value.split('\n')[0])
                    
                    # Colorir baseado na performance
                    if value == best_per_category[cat] and value > 0:
                        # Melhor valor - verde claro
                        row_colors.append('#90EE90')
                    elif value > 0.7:
                        # Bom - azul claro
                        row_colors.append('#ADD8E6')
                    elif value > 0.5:
                        # Médio - amarelo claro
                        row_colors.append('#FFFFE0')
                    else:
                        # Baixo - vermelho claro
                        row_colors.append('#FFB6C1')
                else:
                    row_colors.append('white')
            
            cell_colors.append(row_colors)
        
        # Criar tabela
        col_labels = ['Modelo'] + categories
        table = ax.table(cellText=table_data, 
                        colLabels=col_labels,
                        cellColours=cell_colors,
                        cellLoc='center', 
                        loc='center',
                        colWidths=[0.3] + [0.14]*len(categories))
        
        # Estilizar tabela
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Cabeçalho em azul
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(color='white', weight='bold')
        
        # Título e notas
        plt.title('Métricas Agregadas por Categoria (Dataset Completo)', 
                fontsize=14, fontweight='bold', pad=20)
        
        # Adicionar legenda de cores
        legend_text = ("Cores: Verde=Melhor | Azul=IoU>0.7 | Amarelo=IoU>0.5 | Rosa=IoU≤0.5\n"
                    "Valores: média (n patches) | Clear/Thick/Thin/Shadow mostram IoU, Mixed mostra Accuracy")
        plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Imprimir resumo textual
        print("\n" + "="*100)
        print("TABELA DE MÉTRICAS AGREGADAS POR CATEGORIA")
        print("="*100)
        
        # Header
        header = f"{'Modelo':<35}"
        for cat in categories:
            header += f"{cat:<15}"
        print(header)
        print("-"*100)
        
        # Dados
        for model_name in models_with_metrics:
            metrics = self.results[model_name].additional_info['aggregated_metrics_by_category']
            display_name = model_name[:33] + '..' if len(model_name) > 35 else model_name
            row = f"{display_name:<35}"
            
            for cat in categories:
                full_cat = cat if cat != 'Mixed/Trans.' else 'Mixed/Transition'
                
                if full_cat in metrics and metrics[full_cat]['count'] > 0:
                    mean_val = metrics[full_cat]['mean']
                    count = metrics[full_cat]['count']
                    row += f"{mean_val:.3f} ({count:3d})  "
                else:
                    row += f"---   (  0)  "
            
            print(row)
        
        print("-"*100)
        print("Nota: Clear/Thick/Thin/Shadow mostram IoU médio, Mixed/Transition mostra Accuracy média")

    def plot_error_difference_maps(self, models=None, figsize=(20, 12), save_path=None):
        """
        Versão melhorada: usa múltiplos exemplos e mostra estatísticas detalhadas.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        import matplotlib.gridspec as gridspec
        
        if models is None:
            models = sorted(self.results.keys())[:3]  # Limita a 3 modelos
        
        models_with_examples = []
        for model in models:
            if (model in self.results and 
                hasattr(self.results[model], 'additional_info') and
                'qualitative_examples' in self.results[model].additional_info):
                models_with_examples.append(model)
        
        if not models_with_examples:
            print("Nenhum modelo com exemplos disponíveis.")
            return
        
        # Configurar figura com GridSpec para layout mais flexível
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, len(models_with_examples), figure=fig, 
                            height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.2)
        
        # Cores
        diff_colors = ['#4CAF50', '#FFC107', '#F44336']  # Verde=correto, Amarelo=incerto, Vermelho=erro
        diff_cmap = ListedColormap(diff_colors[:2])  # Usar só verde e vermelho para o mapa binário
        
        class_colors = ['#4CAF50', '#F44336', '#FF9800', '#9C27B0']
        class_cmap = ListedColormap(class_colors)
        
        # Estatísticas agregadas
        all_stats = {}
        
        for model_idx, model_name in enumerate(models_with_examples):
            examples = self.results[model_name].additional_info['qualitative_examples']
            
            # Selecionar múltiplos exemplos para análise
            selected_examples = []
            
            # Pegar exemplos de diferentes categorias
            if examples['typical_errors']:
                selected_examples.extend(examples['typical_errors'][:2])
            if examples['ambiguous_cases']:
                selected_examples.extend(examples['ambiguous_cases'][:2])
            if 'edge_cases' in examples and examples['edge_cases']:
                selected_examples.extend(examples['edge_cases'][:1])
            
            if not selected_examples:
                continue
            
            # Agregar estatísticas de erro
            total_errors = []
            class_specific_errors = {i: [] for i in range(4)}
            
            for example in selected_examples:
                true_mask = np.squeeze(example['true_mask'])
                pred_mask = np.squeeze(example['pred_mask'])
                diff_map = (true_mask != pred_mask).astype(float)
                
                total_errors.append(diff_map.mean())
                
                for class_idx in range(4):
                    class_mask = (true_mask == class_idx)
                    if class_mask.any():
                        class_error = diff_map[class_mask].mean()
                        class_specific_errors[class_idx].append(class_error)
            
            # Escolher o exemplo mais representativo (erro mediano)
            median_error = np.median(total_errors)
            best_example_idx = np.argmin(np.abs(np.array(total_errors) - median_error))
            example = selected_examples[best_example_idx]
            
            true_mask = np.squeeze(example['true_mask'])
            pred_mask = np.squeeze(example['pred_mask'])
            diff_map = (true_mask != pred_mask).astype(int)
            
            # Plot predição
            ax1 = fig.add_subplot(gs[0, model_idx])
            ax1.imshow(pred_mask, cmap=class_cmap, vmin=0, vmax=3)
            ax1.set_title(f'{model_name[:25]}\nPredição', fontsize=10)
            ax1.axis('off')
            
            # Plot mapa de diferença
            ax2 = fig.add_subplot(gs[1, model_idx])
            im = ax2.imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=1)
            error_rate = diff_map.mean() * 100
            ax2.set_title(f'Mapa de Erros\n({error_rate:.1f}% pixels errados)', fontsize=10)
            ax2.axis('off')
            
            # Adicionar contornos nas regiões de erro
            from matplotlib import patches
            # Identificar regiões de erro contíguas (simplificado)
            if diff_map.any():
                y_err, x_err = np.where(diff_map > 0)
                if len(x_err) > 0:
                    # Desenhar pequenos círculos nos erros mais significativos
                    for i in range(min(5, len(x_err))):
                        circle = patches.Circle((x_err[i], y_err[i]), radius=2, 
                                            linewidth=1, edgecolor='yellow', 
                                            facecolor='none', alpha=0.7)
                        ax2.add_patch(circle)
            
            # Plot estatísticas detalhadas
            ax3 = fig.add_subplot(gs[2, model_idx])
            ax3.axis('off')
            
            # Calcular estatísticas por classe
            stats_text = f"Estatísticas de Erro:\n"
            stats_text += f"Total: {error_rate:.1f}%\n\n"
            stats_text += "Por classe:\n"
            
            for class_idx in range(4):
                if class_idx in class_specific_errors and class_specific_errors[class_idx]:
                    mean_error = np.mean(class_specific_errors[class_idx]) * 100
                    stats_text += f"{CLASS_NAMES[class_idx]}: {mean_error:.1f}%\n"
            
            # Adicionar informações sobre confusões
            if 'confusion_pairs' in examples and examples['confusion_pairs']:
                stats_text += "\nConfusões principais:\n"
                for key in list(examples['confusion_pairs'].keys())[:2]:
                    if examples['confusion_pairs'][key]:
                        conf_rate = examples['confusion_pairs'][key][0].get('confusion_rate', 0)
                        stats_text += f"{key.replace('_to_', '→')}: {conf_rate:.1%}\n"
            
            ax3.text(0.5, 0.5, stats_text, ha='center', va='center', 
                    fontsize=8, transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Guardar estatísticas
            all_stats[model_name] = {
                'total_error': error_rate,
                'class_errors': {CLASS_NAMES[i]: np.mean(class_specific_errors[i])*100 
                            if class_specific_errors[i] else 0 
                            for i in range(4)}
            }
        
        # Adicionar colorbar
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.4, 0.02, 0.3])
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=diff_cmap), 
                        cax=cbar_ax, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Correto', 'Erro'], fontsize=9)
        
        # Título geral
        fig.suptitle('Análise Espacial de Erros - Comparação entre Modelos', 
                    fontsize=14, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Imprimir resumo de estatísticas
        print("\n" + "="*60)
        print("RESUMO DE ERROS POR MODELO")
        print("="*60)
        for model, stats in all_stats.items():
            print(f"\n{model}:")
            print(f"  Erro total: {stats['total_error']:.2f}%")
            print("  Erros por classe:")
            for class_name, error in stats['class_errors'].items():
                if error > 0:
                    print(f"    {class_name}: {error:.2f}%")


    def plot_inference_cost(
            self,
            metric: str = "latency_ms",   # 'latency_ms' | 'throughput_ps' | 'peak_mem_mb'
            models: list[str] | None = None,
            figsize: tuple = (10, 6),
            save_path: str | None = None,
        ):
        """
        Gráfico de barras para custo computacional por modelo.
        Barras de erro = quantis p5–p95 (assimétricas).
        Procura primeiro em compute_cost['summary'][key] e, se faltar,
        calcula a partir de compute_cost['per_run'].

        Para memória, prefere 'peak_mem_reserved_mb' (mais estável).
        """
        from matplotlib.patches import Patch
        import numpy as np
        import math
        import matplotlib.pyplot as plt

        if models is None:
            models = sorted(self.results.keys())

        # Mapeia métrica -> chaves esperadas
        if metric == "latency_ms":
            title = "Custo Computacional – Latência"
            ylab  = "ms por patch"
            fmt   = lambda x: f"{x:.1f}"
            key_summary = key_perrun = key_legacy = "latency_ms_per_patch"
            lower_is_better = True
        elif metric == "throughput_ps":
            title = "Custo Computacional – Vazão"
            ylab  = "patches / s"
            fmt   = lambda x: f"{x:.1f}"
            key_summary = key_perrun = key_legacy = "throughput_patches_per_s"
            lower_is_better = False
        elif metric == "peak_mem_mb":
            title = "Custo Computacional – Memória Pico (GPU)"
            ylab  = "MB"
            fmt   = lambda x: f"{x:.0f}"
            # Preferimos 'reserved' (summary/per_run); recuamos p/ legados
            key_summary = "peak_mem_reserved_mb"
            key_perrun  = "peak_mem_reserved_mb"
            key_legacy  = "peak_mem_mb"  # legado (alocado)
            lower_is_better = True
        else:
            raise ValueError("metric deve ser 'latency_ms', 'throughput_ps' ou 'peak_mem_mb'.")

        centers, err_low, err_high, used_models = [], [], [], []

        for m in models:
            res  = self.results.get(m)
            info = getattr(res, "additional_info", {}) or {}
            cc   = info.get("compute_cost", {})
            if not cc:
                continue

            # ---- 1) Tenta summary: median, p5, p95
            summ  = cc.get("summary", {}) or {}
            block = summ.get(key_summary, {})
            med   = block.get("median", None)
            p5    = block.get("p5", None)
            p95   = block.get("p95", None)

            # ---- 2) Fallback: valor legado único (sem dispersão)
            if med is None:
                v = cc.get(key_legacy, None) if metric == "peak_mem_mb" else cc.get(key_summary, None)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    med = float(v)

            # ---- 3) Se faltar p5/p95 (ou até a mediana), calcula de per_run
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

            # ---- 4) Consolida
            if med is None or (isinstance(med, float) and math.isnan(med)):
                continue

            # Erros assimétricos: med - p5  /  p95 - med
            e_low  = float(med - p5)  if (p5  is not None) else 0.0
            e_high = float(p95 - med) if (p95 is not None) else 0.0
            e_low  = max(0.0, e_low)
            e_high = max(0.0, e_high)

            centers.append(float(med))
            err_low.append(e_low)
            err_high.append(e_high)
            used_models.append(m)

        if not used_models:
            raise ValueError("Nenhum modelo possui dados de 'compute_cost'. Execute a avaliação com benchmark.")

        # Índice do "melhor" (lat/mem: menor; throughput: maior)
        import numpy as np
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

        # Removido: nota "Centro = mediana; faixas = p5–p95"

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    
    def plot_efficiency_bubble(self, models=None, figsize=(12, 8), save_path=None):
        """
        Gráfico de bolha comparando eficiência dos modelos.
        
        - Eixo X: Quantidade de parâmetros
        - Eixo Y: IoU médio (média das 4 classes)
        - Tamanho da bolha: GFLOPS
        
        Parameters
        ----------
        models : list[str], optional
            Lista de modelos para comparar. Se None, usa todos.
        figsize : tuple
            Tamanho da figura.
        save_path : str, optional
            Caminho para salvar o PNG se desejado.
        """
        from matplotlib.patches import Patch
        import numpy as np
        import matplotlib.pyplot as plt
        
        if models is None:
            models = sorted(self.results.keys())
        
        # Coletar dados
        params_list = []
        iou_list = []
        gflops_list = []
        used_models = []
        
        for m in models:
            res = self.results.get(m)
            if not res:
                continue
                
            info = getattr(res, "additional_info", {}) or {}
            
            # Validar dados necessários
            if 'n_parameters' not in info:
                print(f"Aviso: {m} não possui 'n_parameters'. Pulando...")
                continue
            if 'gflops' not in info:
                print(f"Aviso: {m} não possui 'gflops'. Pulando...")
                continue
            
            # Calcular IoU médio das 4 classes
            metrics = res.metrics
            ious = []
            for class_name in CLASS_NAMES:
                if class_name in metrics and 'F1-Score' in metrics[class_name]:
                    # Usar F1-Score como proxy se IoU não estiver disponível
                    # ou buscar IoU diretamente se disponível
                    if 'IoU' in metrics[class_name]:
                        ious.append(metrics[class_name]['IoU'])
                    else:
                        # F1-Score é uma boa aproximação quando IoU não está disponível
                        ious.append(metrics[class_name]['F1-Score'])
            
            if not ious:
                print(f"Aviso: {m} não possui métricas de IoU/F1. Pulando...")
                continue
            
            mean_iou = np.mean(ious)
            
            params_list.append(info['n_parameters'])
            iou_list.append(mean_iou)
            gflops_list.append(info['gflops'])
            used_models.append(m)
        
        if not used_models:
            raise ValueError("Nenhum modelo possui dados completos (n_parameters, gflops, métricas).")
        
        # Normalizar tamanho das bolhas (entre 100 e 2000 para boa visualização)
        gflops_array = np.array(gflops_list)
        min_gflops = gflops_array.min()
        max_gflops = gflops_array.max()
        
        if max_gflops == min_gflops:
            bubble_sizes = [500] * len(gflops_list)
        else:
            # Escala logarítmica para melhor diferenciação
            bubble_sizes = 100 + 1900 * (np.log1p(gflops_array - min_gflops) / 
                                        np.log1p(max_gflops - min_gflops))
        
        # Identificar melhor modelo (maior IoU médio)
        best_idx = int(np.argmax(iou_list))
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot das bolhas
        for i, m in enumerate(used_models):
            color = self._get_color(m)
            
            # Scatter plot
            scatter = ax.scatter(params_list[i], iou_list[i], 
                               s=bubble_sizes[i], 
                               c=[color], 
                               alpha=0.6,
                               edgecolors='black' if i == best_idx else 'gray',
                               linewidths=2.5 if i == best_idx else 1,
                               linestyle='--' if i == best_idx else '-')
            
            # Anotar com nome do modelo
            ax.annotate(m, 
                       (params_list[i], iou_list[i]),
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', 
                               alpha=0.7,
                               edgecolor=color))
        
        # Configurar eixos
        ax.set_xlabel('Número de Parâmetros', fontsize=12)
        ax.set_ylabel('IoU Médio', fontsize=12)
        ax.set_title('Eficiência dos Modelos: IoU vs Parâmetros vs GFLOPS', 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Adicionar margem aos limites
        x_range = max(params_list) - min(params_list)
        y_range = max(iou_list) - min(iou_list)
        ax.set_xlim(min(params_list) - 0.1*x_range, max(params_list) + 0.1*x_range)
        ax.set_ylim(min(iou_list) - 0.05*y_range, max(iou_list) + 0.05*y_range)
        
        # Legenda de tamanhos
        # Criar bolhas de referência para a legenda
        legend_gflops = [min_gflops, (min_gflops + max_gflops)/2, max_gflops]
        legend_sizes = []
        for g in legend_gflops:
            if max_gflops == min_gflops:
                legend_sizes.append(500)
            else:
                s = 100 + 1900 * (np.log1p(g - min_gflops) / 
                                 np.log1p(max_gflops - min_gflops))
                legend_sizes.append(s)
        
        # Adicionar legenda de tamanhos no canto
        legend_x = ax.get_xlim()[1] * 0.85
        legend_y_base = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15
        
        for i, (gf, sz) in enumerate(zip(legend_gflops, legend_sizes)):
            y_offset = i * 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.scatter(legend_x, legend_y_base + y_offset, 
                      s=sz, c='gray', alpha=0.4, edgecolors='black', linewidths=0.5)
            ax.text(legend_x * 1.05, legend_y_base + y_offset, 
                   f'{gf:.1f} GFLOPS', 
                   fontsize=8, va='center')
        
        # Adicionar nota sobre o melhor modelo
        best_model = used_models[best_idx]
        note = (f"★ Melhor IoU: {best_model}\n"
                f"   IoU={iou_list[best_idx]:.3f}, "
                f"Params={params_list[best_idx]:,.0f}, "
                f"GFLOPS={gflops_list[best_idx]:.1f}")
        ax.text(0.02, 0.98, note, 
               transform=ax.transAxes,
               fontsize=9,
               va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Imprimir tabela resumo
        print("\n" + "="*80)
        print("RESUMO DE EFICIÊNCIA DOS MODELOS")
        print("="*80)
        print(f"{'Modelo':<35} {'Parâmetros':>12} {'IoU Médio':>12} {'GFLOPS':>10}")
        print("-"*80)
        
        # Ordenar por IoU (melhor primeiro)
        sorted_indices = np.argsort(iou_list)[::-1]
        for idx in sorted_indices:
            marker = "★ " if idx == best_idx else "  "
            print(f"{marker}{used_models[idx]:<33} {params_list[idx]:>12,.0f} "
                  f"{iou_list[idx]:>12.4f} {gflops_list[idx]:>10.2f}")
        print("="*80)