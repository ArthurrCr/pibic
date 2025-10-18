import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import patches
from typing import List, Optional, Tuple, Dict, Any

from .plot_utils import CLASS_NAMES, CLASS_COLORS


class TablePlotter:
    """Cria tabelas e visualizações agregadas."""
    
    def __init__(self, results_manager):
        # Guardamos o manager para eventuais leituras futuras; hoje usamos apenas .results
        self._rm = results_manager
        self.results = results_manager.results

    # ------------------------------------------------------------------ #
    #                       TABELA AGREGADA (IoU/Acc)                     #
    # ------------------------------------------------------------------ #
    def plot_aggregated_metrics_table(
        self,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ) -> None:
        """Plota tabela comparativa das métricas agregadas por categoria."""
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
            # Alinha com o monolítico: ajuda sobre fluxo de avaliação antes de plotar
            print("Execute evaluate_model() primeiro.")
            return
        
        # Preparar dados para a tabela
        categories = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Shadow', 'Mixed/Trans.']
        table_data = []
        
        # Rastrear melhor valor por categoria (para colorir)
        best_per_category = {cat: 0.0 for cat in categories}
        
        for model_name in models_with_metrics:
            metrics = self.results[model_name].additional_info['aggregated_metrics_by_category']
            
            # Truncar nome se necessário
            display_name = model_name[:30] + '..' if len(model_name) > 32 else model_name
            
            row = [display_name]
            for cat in categories:
                # Nome canônico para busca
                full_cat = cat if cat != 'Mixed/Trans.' else 'Mixed/Transition'
                
                if full_cat in metrics and metrics[full_cat].get('count', 0) > 0:
                    mean_val = float(metrics[full_cat]['mean'])
                    count = int(metrics[full_cat]['count'])
                    row.append(f"{mean_val:.3f}\n({count})")
                    # Atualiza “melhor por categoria”
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
        for row in table_data:
            row_colors = ['#f0f0f0']  # Coluna do nome do modelo
            for col_idx, cat in enumerate(categories):
                cell_value = row[col_idx + 1]
                if '---' not in cell_value:
                    value = float(cell_value.split('\n')[0])
                    # Regras de cor (verde = melhor, azul/amar/amarelo/rosa conforme faixas)
                    if value == best_per_category[cat] and value > 0:
                        row_colors.append('#90EE90')   # Verde claro
                    elif value > 0.7:
                        row_colors.append('#ADD8E6')   # Azul claro
                    elif value > 0.5:
                        row_colors.append('#FFFFE0')   # Amarelo claro
                    else:
                        row_colors.append('#FFB6C1')   # Rosa claro
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
        
        # Legenda de cores
        legend_text = ("Cores: Verde=Melhor | Azul=IoU>0.7 | Amarelo=IoU>0.5 | Rosa=IoU≤0.5\n"
                       "Valores: média (n patches) | Clear/Thick/Thin/Shadow mostram IoU, Mixed mostra Accuracy")
        plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Resumo textual (igual ao fluxo do arquivo modular anterior)
        self._print_aggregated_summary(models_with_metrics, categories)

    # ------------------------------------------------------------------ #
    #                 MAPAS DE DIFERENÇA / ESTATÍSTICAS                  #
    # ------------------------------------------------------------------ #
    def plot_error_difference_maps(
        self,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 12),
        save_path: Optional[str] = None
    ) -> None:
        """
        Compara mapas de erro entre modelos usando exemplos representativos.
        Aceita 'qualitative_examples' como LISTA ou como DICIONÁRIO com
        chaves ('typical_errors', 'ambiguous_cases', 'edge_cases').
        """
        if models is None:
            models = sorted(self.results.keys())[:3]  # Limita a 3 modelos por legibilidade
        
        models_with_examples = []
        for model in models:
            if (model in self.results and 
                hasattr(self.results[model], 'additional_info') and
                'qualitative_examples' in self.results[model].additional_info):
                examples = self.results[model].additional_info['qualitative_examples']
                if (isinstance(examples, list) and len(examples) > 0) or \
                   (isinstance(examples, dict) and any(bool(examples.get(k)) for k in
                                                       ('typical_errors', 'ambiguous_cases', 'edge_cases'))):
                    models_with_examples.append(model)
        
        if not models_with_examples:
            print("Nenhum modelo com exemplos disponíveis.")
            return
        
        # Configurar figura com GridSpec
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, len(models_with_examples), figure=fig, 
                               height_ratios=[1, 1, 0.35], hspace=0.3, wspace=0.2)
        
        # Cores
        # Mapa para classes (segue CLASS_COLORS)
        class_colors = [CLASS_COLORS[i] for i in range(4)]
        class_cmap = ListedColormap(class_colors)
        # Mapa binário para correto/erro (verde -> correto, vermelho -> erro)
        diff_cmap = ListedColormap(['#4CAF50', '#F44336'])
        
        # Estatísticas agregadas
        all_stats: Dict[str, Dict[str, Any]] = {}
        
        for col_idx, model_name in enumerate(models_with_examples):
            raw = self.results[model_name].additional_info['qualitative_examples']
            
            # Normaliza para uma lista de exemplos
            examples_list = self._normalize_examples(raw)
            if not examples_list:
                continue
            
            # Selecione um conjunto curto e representativo:
            # calculamos o erro por exemplo e pegamos o "mediano"
            total_errors = []
            for ex in examples_list:
                true_mask = np.squeeze(ex['true_mask'])
                pred_mask = np.squeeze(ex['pred_mask'])
                diff_map = (true_mask != pred_mask).astype(np.float32)
                total_errors.append(diff_map.mean())
            total_errors = np.array(total_errors)
            median_error = np.median(total_errors)
            best_example_idx = int(np.argmin(np.abs(total_errors - median_error)))
            example = examples_list[best_example_idx]

            # Mapas do exemplo representativo
            true_mask = np.squeeze(example['true_mask'])
            pred_mask = np.squeeze(example['pred_mask'])
            diff_map = (true_mask != pred_mask).astype(int)
            
            # ---- Linha 0: Predição por classe
            ax1 = fig.add_subplot(gs[0, col_idx])
            ax1.imshow(pred_mask, cmap=class_cmap, vmin=0, vmax=3)
            ax1.set_title(f'{model_name[:25]}\nPredição', fontsize=10)
            ax1.axis('off')
            
            # ---- Linha 1: Mapa de diferença (0=correto, 1=erro)
            ax2 = fig.add_subplot(gs[1, col_idx])
            ax2.imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=1)
            error_rate = diff_map.mean() * 100
            ax2.set_title(f'Mapa de Erros\n({error_rate:.1f}% pixels errados)', fontsize=10)
            ax2.axis('off')
            
            # Pequenos realces de erro (círculos)
            if diff_map.any():
                y_err, x_err = np.where(diff_map > 0)
                if len(x_err) > 0:
                    for i in range(min(5, len(x_err))):
                        circle = patches.Circle((x_err[i], y_err[i]), radius=2,
                                                linewidth=1, edgecolor='yellow',
                                                facecolor='none', alpha=0.7)
                        ax2.add_patch(circle)
            
            # ---- Linha 2: Estatísticas textuais
            ax3 = fig.add_subplot(gs[2, col_idx])
            ax3.axis('off')
            
            # Estatísticas por classe usando todos os exemplos (mais estável)
            class_specific_errors = {i: [] for i in range(4)}
            for ex in examples_list:
                t = np.squeeze(ex['true_mask'])
                p = np.squeeze(ex['pred_mask'])
                d = (t != p).astype(np.float32)
                for c in range(4):
                    cmask = (t == c)
                    if cmask.any():
                        class_specific_errors[c].append(d[cmask].mean())
            
            stats_text = f"Estatísticas de Erro:\n"
            stats_text += f"Total (ex. rep.): {error_rate:.1f}%\n\n"
            stats_text += "Por classe (média):\n"
            for c in range(4):
                vals = class_specific_errors[c]
                if vals:
                    stats_text += f"{CLASS_NAMES[c]}: {np.mean(vals) * 100:.1f}%\n"
            
            # Se houver pares de confusão agregados no dicionário original, mostre-os
            if isinstance(raw, dict) and raw.get('confusion_pairs'):
                pairs = raw['confusion_pairs']
                top_keys = list(pairs.keys())[:2]
                if top_keys:
                    stats_text += "\nConfusões principais:\n"
                    for k in top_keys:
                        if pairs[k]:
                            conf_rate = pairs[k][0].get('confusion_rate', 0.0)
                            stats_text += f"{k.replace('_to_', '→')}: {conf_rate:.1%}\n"
            
            ax3.text(0.5, 0.5, stats_text, ha='center', va='center',
                     fontsize=8, transform=ax3.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Guardar estatísticas para o resumo textual
            all_stats[model_name] = {
                'total_error': error_rate,
                'class_errors': {
                    CLASS_NAMES[i]: (np.mean(class_specific_errors[i]) * 100) if class_specific_errors[i] else 0.0
                    for i in range(4)
                }
            }
        
        # Colorbar binária (0/1)
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
        
        # Resumo textual
        self._print_error_summary(all_stats)

    # ------------------------------------------------------------------ #
    #                          FUNÇÕES AUXILIARES                         #
    # ------------------------------------------------------------------ #
    def _print_aggregated_summary(self, models: List[str], categories: List[str]) -> None:
        """Imprime resumo textual das métricas agregadas."""
        print("\n" + "="*100)
        print("TABELA DE MÉTRICAS AGREGADAS POR CATEGORIA")
        print("="*100)
        
        # Header
        header = f"{'Modelo':<35}" + "".join(f"{cat:<15}" for cat in categories)
        print(header)
        print("-"*100)
        
        # Dados
        for model_name in models:
            metrics = self.results[model_name].additional_info['aggregated_metrics_by_category']
            display_name = model_name[:33] + '..' if len(model_name) > 35 else model_name
            row = f"{display_name:<35}"
            
            for cat in categories:
                full_cat = cat if cat != 'Mixed/Trans.' else 'Mixed/Transition'
                if full_cat in metrics and metrics[full_cat].get('count', 0) > 0:
                    mean_val = float(metrics[full_cat]['mean'])
                    count = int(metrics[full_cat]['count'])
                    row += f"{mean_val:.3f} ({count:3d})  "
                else:
                    row += f"---   (  0)  "
            print(row)
        
        print("-"*100)
        print("Nota: Clear/Thick/Thin/Shadow mostram IoU médio; Mixed/Transition mostra Accuracy média.")

    def _print_error_summary(self, all_stats: Dict[str, Dict[str, Any]]) -> None:
        """Imprime resumo de estatísticas de erro por modelo/classe."""
        print("\n" + "="*60)
        print("RESUMO DE ERROS POR MODELO")
        print("="*60)
        for model, stats in all_stats.items():
            print(f"\n{model}:")
            print(f"  Erro total (exemplo representativo): {stats['total_error']:.2f}%")
            print("  Erros por classe (média):")
            for class_name, error in stats['class_errors'].items():
                if error > 0:
                    print(f"    {class_name}: {error:.2f}%")

    # ------------------------------ Utils ------------------------------ #
    @staticmethod
    def _normalize_examples(raw: Any) -> List[Dict[str, Any]]:
        """
        Converte a estrutura de 'qualitative_examples' em lista de exemplos.
        - Se já for lista, retorna diretamente.
        - Se for dicionário, concatena listas conhecidas na ordem:
          typical_errors (até 2), ambiguous_cases (até 2), edge_cases (até 1),
          mantendo robustez caso alguma chave não exista.
        """
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            pooled: List[Dict[str, Any]] = []
            for key, limit in (('typical_errors', 2), ('ambiguous_cases', 2), ('edge_cases', 1)):
                vals = raw.get(key) or []
                if isinstance(vals, list) and len(vals) > 0:
                    pooled.extend(vals[:limit])
            return pooled
        return []
