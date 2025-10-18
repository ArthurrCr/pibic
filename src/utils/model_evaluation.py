import torch
import pickle
from pathlib import Path
from utils.evaluation import (
    evaluate_clouds2mask,
    compute_metrics,
    plot_confusion_matrix,
    get_normalization_stats,
    get_predictions,
    normalize_images,
    SENTINEL_BANDS
)

from utils.calculate_BOA_metrics import (
    evaluate_test_dataset,
    evaluate_test_dataset_with_thresholds,
    find_optimal_threshold_by_patch
)

CLASS_NAMES = ['Clear','Thick Cloud','Thin Cloud','Cloud Shadow']

class ModelEvaluator:
    def __init__(self, manager, device=None, cache_dir='/content/drive/MyDrive/pibic/evaluation_cache'):
        self.manager = manager
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Usando dispositivo: {self.device}")
        print(f"Cache dir: {self.cache_dir}")
        
        # Configurações específicas por modelo
        self.model_configs = {
            "CloudS2Mask ensemble": {
                "use_ensemble": True,
                "normalize_imgs": True
            }
        }
        
        # Índice de modelos
        self._model_index = {}
        
        # Carregar automaticamente todos os resultados existentes
        self._auto_load_existing_results()
    
    def _auto_load_existing_results(self):
        """Carrega automaticamente todos os resultados de avaliação existentes"""
        print("\nVerificando cache de avaliações...")
        loaded_count = 0
        
        # Criar arquivo de índice para mapear nomes
        index_file = self.cache_dir / "model_index.pkl"
        model_index = {}
        
        # Tentar carregar índice existente
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    model_index = pickle.load(f)
            except:
                model_index = {}
        
        # Carregar todos os arquivos de avaliação
        for file in self.cache_dir.glob("*_evaluation_results.pkl"):
            try:
                with open(file, 'rb') as f:
                    result = pickle.load(f)
                
                # Procurar nome do modelo no índice
                model_name = None
                for name, path in model_index.items():
                    if path == file.name:
                        model_name = name
                        break
                
                # Se não encontrou no índice, tentar extrair do nome do arquivo
                if not model_name:
                    # Tentar recuperar o nome original (melhor esforço)
                    parts = file.stem.split('_')
                    if len(parts) > 2:
                        # Remover hash e tipo
                        model_name = '_'.join(parts[:-2]).replace('_', ' ').replace('-', '/')
                
                if model_name:
                    self.manager.results[model_name] = result
                    loaded_count += 1
                    print(f"  ✓ Carregado: {model_name}")
                    
            except Exception as e:
                print(f"  ✗ Erro ao carregar {file.name}: {e}")
        
        if loaded_count > 0:
            print(f"\n{loaded_count} avaliação(ões) carregada(s) do cache.")
        else:
            print("Nenhuma avaliação anterior encontrada.")
        
        self._model_index = model_index
    
    def _get_cache_path(self, model_name, eval_type):
        """Gera caminho único para cache baseado no nome do modelo e tipo de avaliação"""
        import hashlib
        name_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        safe_name = model_name.replace("/", "-").replace(" ", "_")
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        filename = f"{safe_name}_{name_hash}_{eval_type}.pkl"
        return self.cache_dir / filename
    
    
    def evaluate_model(self, model_name, models, test_loader, val_loader=None, 
                  use_ensemble=None, normalize_imgs=None):
        """
        Avaliação completa de um modelo ou ensemble com cache automático
        """
        
        # Verificar se já existe nos resultados carregados
        if model_name in self.manager.results:
            print(f"\n{'='*60}")
            print(f"RESULTADOS JÁ EXISTEM PARA: {model_name}")
            print(f"{'='*60}")
            print("✓ Usando resultados do cache")
            self._print_cached_summary(model_name)
            self._plot_all_results(model_name)
            return
        
        print(f"\n{'='*60}")
        print(f"INICIANDO AVALIAÇÃO: {model_name}")
        print(f"{'='*60}")
        
        # Usar configuração específica do modelo se disponível
        if model_name in self.model_configs:
            config = self.model_configs[model_name]
            use_ensemble = config["use_ensemble"] if use_ensemble is None else use_ensemble
            normalize_imgs = config["normalize_imgs"] if normalize_imgs is None else normalize_imgs
        else:
            use_ensemble = use_ensemble if use_ensemble is not None else False
            normalize_imgs = normalize_imgs if normalize_imgs is not None else False
        
        print(f"Número de modelos: {len(models)}")
        print(f"Usando ensemble: {use_ensemble}")
        print(f"Normalizar imagens: {normalize_imgs}")
        
        # Matriz de Confusão (TESTE) — calcula e salva (não plota aqui)
        self._evaluate_confusion_matrix(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )
        
        # Avaliação por Patch (TESTE, argmax baseline)
        self._evaluate_patches(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )

        # Coletar exemplos qualitativos para visualização
        self._collect_qualitative_examples(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )
        
        self._compute_aggregated_metrics_by_category(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )
        
        # Limiares Ótimos: CALIBRA NA VALIDAÇÃO e APLICA NO TESTE
        self._calculate_optimal_thresholds(
            model_name, models, test_loader, use_ensemble, normalize_imgs, val_loader=val_loader
        )

        # Resumo Final (mostra BOA no TESTE aplicando t*)
        self._print_summary(model_name)

        # Número de parâmetros (único caminho, funciona p/ 1 ou N modelos)
        n_params = sum(p.numel() for m in models for p in m.parameters())
        # Garante a existência de additional_info
        if not hasattr(self.manager.results[model_name], 'additional_info') or self.manager.results[model_name].additional_info is None:
            self.manager.results[model_name].additional_info = {}
        self.manager.results[model_name].additional_info['n_parameters'] = n_params

        # Benchmark de custo computacional (usa as flags efetivas já resolvidas)
        self._benchmark_inference_costs(
            model_name, models, test_loader,
            use_ensemble=use_ensemble,
            normalize_imgs=normalize_imgs,
            warmup_batches=5,
            measure_batches=None,      # usa critério por amostra/tempo
            repetitions=5,             # 3–5 é o recomendado
            min_patches=500,           # ≥ 500 patches
            min_measured_ms=5000.0,    # ou ≥ 5 s medidos
        )

        # Plots finais (apenas o que queremos: matriz de confusão 1x)
        self._plot_all_results(model_name)

        # Salvar automaticamente estado completo
        self._save_complete_state(model_name)
        
    def _plot_all_results(self, model_name):
        """Plota os gráficos de resultados desejados.
        - NÃO plota a tabela estilizada (removido a pedido).
        - Plota a matriz de confusão apenas aqui (evita duplicidade).
        """
        plot_confusion_matrix(
            self.manager.results[model_name].confusion_matrix, 
            normalize=True,
            title=f'Matriz de Confusão - {model_name}'
        )
    
    def _save_complete_state(self, model_name):
        """Salva apenas as métricas e resultados da avaliação"""
        if model_name in self.manager.results:
            cache_path = self._get_cache_path(model_name, "evaluation_results")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.manager.results[model_name], f)
            
            index_file = self.cache_dir / "model_index.pkl"
            model_index = {}
            if index_file.exists():
                try:
                    with open(index_file, 'rb') as f:
                        model_index = pickle.load(f)
                except:
                    model_index = {}
            model_index[model_name] = cache_path.name
            with open(index_file, 'wb') as f:
                pickle.dump(model_index, f)
            print(f"\n✓ Métricas de avaliação salvas para: {model_name}")
    
    def _print_cached_summary(self, model_name):
        """Imprime resumo dos resultados em cache"""
        result = self.manager.results[model_name]
        print(f"\nAcurácia Global: {result.overall_accuracy:.4f}")
        print(f"Timestamp: {result.timestamp[:19]}")
        
        print("\nMétricas por Classe:")
        print("-"*40)
        for class_name, metrics in result.metrics.items():
            print(f"\n{class_name}:")
            print(f"  F1-Score: {metrics['F1-Score']:.4f}")
            print(f"  Precision: {metrics['Precision']:.4f}")
            print(f"  Recall: {metrics['Recall']:.4f}")
            print(f"  Omission Error: {metrics['Omission Error']:.4f}")
            print(f"  Commission Error: {metrics['Commission Error']:.4f}")
        
        # ✅ Resumo honesto: mostra BOA no TESTE aplicando t* (val-cal)
        print("\n" + "="*60)
        print("RESUMO DOS LIMIARES (val-cal → aplicados no TESTE)")
        print("="*60)

        aux = getattr(result, "additional_info", {}) or {}

        # ⚠️ Coalescência sem 'or' com DataFrame
        df_val_applied = aux.get('df_thresh_valcal')
        if df_val_applied is None:
            df_val_applied = aux.get('df_thresh')

        t_star_valcal = aux.get('t_star_valcal')
        if t_star_valcal is None:
            t_star_valcal = aux.get('t_star')

        if df_val_applied is None or t_star_valcal is None:
            print("\n(sem dados de t* aplicados ao teste no cache atual)")
            return

        def _get(df, exp, col):
            row = df.loc[df['Experiment'] == exp]
            if row.empty: return float('nan')
            v = row[col].iloc[0]
            try: return float(v)
            except: return float(str(v))

        for exp in ["cloud/no cloud", "cloud shadow", "valid/invalid"]:
            base = result.boa_baseline.get(exp, float('nan'))
            boa_applied = _get(df_val_applied, exp, 'Median BOA')
            thr = t_star_valcal.get(exp, None)
            thr_str = f"{thr:.2f}" if isinstance(thr, (int,float)) else "?"
            print(f"\n{exp}:")
            print(f"  BOA (argmax, TESTE): {base:.4f}")
            print(f"  t* (calibrado na VAL): {thr_str} | BOA (aplicada no TESTE): {boa_applied:.4f} | Δ: {(boa_applied - base):+.4f}")
    
    def _evaluate_confusion_matrix(self, model_name, models, test_loader, 
                                  use_ensemble, normalize_imgs):
        print("\n" + "="*60)
        print("AVALIAÇÃO 1: MÉTRICAS GERAIS E MATRIZ DE CONFUSÃO")
        print("="*60)
        
        conf_matrix = evaluate_clouds2mask(
            test_loader, models, self.device,
            use_ensemble=use_ensemble,
            normalize_imgs=normalize_imgs
        )
        
        metrics = compute_metrics(conf_matrix)
        self.manager.parse_metrics_from_output(model_name, metrics, conf_matrix)
        
        # 🔵 Não plota aqui para evitar duplicidade; o plot ocorre em _plot_all_results
        self._print_metrics(metrics)
    
    def _evaluate_patches(self, model_name, models, test_loader, 
                         use_ensemble, normalize_imgs):
        print("\n" + "="*60)
        print("AVALIAÇÃO 2: POR PATCH")
        print("="*60)
        
        df_results = evaluate_test_dataset(
            test_loader, models, self.device,
            use_ensemble=use_ensemble,
            normalize_imgs=normalize_imgs
        )
        
        print("\nTabela de Resultados:")
        print("-"*100)
        print(df_results.to_string(index=False))
        
        self.manager.save_boa_results(model_name, df_results=df_results)
    
    def _calculate_optimal_thresholds(self, model_name, models, test_loader,
                                  use_ensemble, normalize_imgs, val_loader=None):
        print("\n" + "="*60)
        print("CALCULANDO LIMIAR ÓTIMO (t*) POR PATCH")
        print("="*60)

        if val_loader is None:
            raise ValueError("val_loader é obrigatório para calibrar t* na validação.")

        experiments = ['cloud/no cloud', 'cloud shadow', 'valid/invalid']

        # 1) Calibração na VALIDAÇÃO
        t_star_valcal = {}
        thr_results_valcal = {}

        print("\n--- Calibração de t* na VALIDAÇÃO (avaliado no TESTE depois) ---")
        for experiment in experiments:
            print(f"· t* (val-cal) para {experiment}...")
            res_val = find_optimal_threshold_by_patch(
                test_loader=val_loader,  # calibra na validação
                models=models,
                experiment=experiment,
                device=self.device,
                use_ensemble=use_ensemble,
                normalize_imgs=normalize_imgs
            )
            res_val['calibrated_on'] = 'val'
            thr_results_valcal[experiment] = res_val
            t_star_valcal[experiment] = res_val['best_threshold']

        # 2) Avalia no TESTE usando t* calibrado na VALIDAÇÃO
        df_thresh_valcal = evaluate_test_dataset_with_thresholds(
            test_loader,
            models,
            t_star_valcal,
            device=str(self.device),
            use_ensemble=use_ensemble,
            normalize_imgs=normalize_imgs
        )

        # 3) Salvar os resultados da BUSCA DE LIMIAR (feita na VALIDAÇÃO)
        #    em results[...].optimal_thresholds[exp], pois o manager usa
        #    este campo para (eventuais) curvas e reabrir do cache.
        for experiment in experiments:
            self.manager.save_boa_results(
                model_name,
                threshold_results=thr_results_valcal[experiment],  # contém best_threshold e median_boa da VAL
                experiment=experiment
            )

        # 4) Guardar no additional_info
        result_obj = self.manager.results[model_name]
        aux = result_obj.additional_info if hasattr(result_obj, 'additional_info') and result_obj.additional_info is not None else {}
        aux.update({
            'thresholds_source_canonical': 'val',  # explícito
            't_star_valcal': t_star_valcal,        # dicionário {exp: t*}
            'thr_results_valcal': thr_results_valcal,  # saída completa da calibração (VAL)
            'df_thresh_valcal': df_thresh_valcal,  # DataFrame: TESTE aplicando t* (VAL)
            # aliases convenientes
            't_star': t_star_valcal,
            'df_thresh': df_thresh_valcal
        })
        result_obj.additional_info = aux

        # 5) Impressão-resumo (TESTE aplicando t* calibrado na VAL)
        print("\n========== RESUMO t* (VAL → TESTE) ==========")
        for exp in experiments:
            base_boa = result_obj.boa_baseline.get(exp, float('nan'))
            # BOA aplicada no TESTE:
            try:
                boa_aplicada = float(
                    df_thresh_valcal.loc[df_thresh_valcal['Experiment'] == exp, 'Median BOA'].iloc[0]
                )
            except Exception:
                boa_aplicada = float('nan')

            bv = thr_results_valcal[exp]['best_threshold']
            print(f"\n[{exp}]")
            print(f"  BOA (argmax, TESTE): {base_boa:.4f}")
            print(f"  t* (calibrado na VAL): {bv:.2f} | BOA aplicada (TESTE): {boa_aplicada:.4f} | Δ: {(boa_aplicada - base_boa):+.4f}")
    
    def _print_metrics(self, metrics):
        print("\nRESULTADOS DA AVALIAÇÃO:")
        print("-"*40)
        
        for class_name, class_metrics in metrics.items():
            if class_name != 'Overall':
                print(f"\n{class_name}:")
                for metric_name, value in class_metrics.items():
                    if metric_name != 'Support':
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {int(value)}")
        
        print(f"\nAcurácia Geral: {metrics['Overall']['Accuracy']:.4f}")
        print(f"Total de Amostras: {int(metrics['Overall']['Total Samples'])}")

    
    def _print_summary(self, model_name):
        print("\n" + "="*60)
        print("RESUMO DOS LIMIARES (val-cal → aplicados no TESTE)")
        print("="*60)
        
        model_result = self.manager.results[model_name]
        aux = getattr(model_result, "additional_info", {}) or {}

        # ⚠️ Coalescência sem 'or' com DataFrame
        df_val_applied = aux.get('df_thresh_valcal')
        if df_val_applied is None:
            df_val_applied = aux.get('df_thresh')

        t_star_valcal = aux.get('t_star_valcal')
        if t_star_valcal is None:
            t_star_valcal = aux.get('t_star')

        if df_val_applied is None or t_star_valcal is None:
            print("\n(sem dados de t* aplicados ao teste)")
            return

        def _get(df, exp, col):
            row = df.loc[df['Experiment'] == exp]
            if row.empty: return float('nan')
            v = row[col].iloc[0]
            try: return float(v)
            except: return float(str(v))

        for exp in ["cloud/no cloud", "cloud shadow", "valid/invalid"]:
            baseline = model_result.boa_baseline.get(exp, float('nan'))
            boa_applied = _get(df_val_applied, exp, 'Median BOA')
            thr = t_star_valcal.get(exp, None)
            thr_str = f"{thr:.2f}" if isinstance(thr, (int,float)) else "?"
            
            print(f"\n{exp}:")
            print(f"  BOA (argmax, TESTE): {baseline:.4f}")
            print(f"  BOA (t*={thr_str}, aplic. no TESTE): {boa_applied:.4f}")
            print(f"  Melhoria: {(boa_applied - baseline):+.4f}")


    def _collect_qualitative_examples(self, model_name, model, test_loader):
        """
        Coleta estratégica de exemplos usando critérios baseados em quantis do dataset
        """
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        print("\n" + "="*60)
        print("COLETANDO EXEMPLOS QUALITATIVOS ESTRATÉGICOS")
        print("="*60)
        
        if not hasattr(self, 'qualitative_indices') or self.qualitative_indices is None:
            print("Analisando TODO o conjunto de teste para seleção baseada em quantis...")
            
            all_patches_info = []
            
            model.eval()
            
            with torch.no_grad():
                for batch_idx, (images, masks) in enumerate(test_loader):
                    images = images.to(self.device).float()
                    masks_cpu = masks.cpu()
                    
                    # Predição com modelo único
                    output = model(images)
                    avg_prediction = F.softmax(output, dim=1)
                    
                    # Entropy com clamp_min
                    entropy = -(avg_prediction * torch.log(avg_prediction.clamp_min(1e-8))).sum(dim=1)  # [B,H,W]
                    
                    for img_idx in range(masks_cpu.size(0)):
                        mask = masks_cpu[img_idx]
                        unique_classes = mask.unique().tolist()
                        
                        total_pixels = mask.numel()
                        fractions = {}
                        for c in range(4):
                            fractions[c] = (mask == c).sum().item() / total_pixels
                        
                        # Boundary density sem wrap-around
                        bd_h = (mask[:, 1:] != mask[:, :-1]).float().sum()
                        bd_v = (mask[1:, :] != mask[:-1, :]).float().sum()
                        boundary_density = (bd_h + bd_v).item() / (2.0 * mask.numel())
                        
                        patch_info = {
                            'batch_idx': batch_idx,
                            'image_idx': img_idx,
                            'classes': unique_classes,
                            'fractions': fractions,
                            'entropy': entropy[img_idx].mean().cpu().item(),
                            'boundary_density': boundary_density,
                            'n_classes': len(unique_classes)
                        }
                        all_patches_info.append(patch_info)
            
            print(f"Total de patches analisados: {len(all_patches_info)}")
            
            # Calcular quantis das distribuições
            fractions_by_class = {c: [] for c in range(4)}
            boundary_densities = []
            entropy_values = []
            
            for patch in all_patches_info:
                for c in range(4):
                    if patch['fractions'][c] > 0:
                        fractions_by_class[c].append(patch['fractions'][c])
                boundary_densities.append(patch['boundary_density'])
                entropy_values.append(patch['entropy'])
            
            quantiles = {}
            for c in range(4):
                if fractions_by_class[c]:
                    quantiles[c] = {
                        'Q50': np.percentile(fractions_by_class[c], 50),
                        'Q75': np.percentile(fractions_by_class[c], 75),
                        'Q80': np.percentile(fractions_by_class[c], 80),
                        'Q90': np.percentile(fractions_by_class[c], 90)
                    }
            
            # Quantis para boundary density
            boundary_quantiles = {
                'Q50': np.percentile(boundary_densities, 50),
                'Q75': np.percentile(boundary_densities, 75),
                'Q80': np.percentile(boundary_densities, 80),
                'Q90': np.percentile(boundary_densities, 90)
            }
            
            print("\nDistribuição de frações por classe (para patches que contêm a classe):")
            for c in range(4):
                if c in quantiles:
                    q = quantiles[c]
                    print(f"  {CLASS_NAMES[c]}: Q50={q['Q50']:.2f}, Q75={q['Q75']:.2f}, Q90={q['Q90']:.2f}")
            
            print(f"\nDistribuição de densidade de fronteiras:")
            print(f"  Q50={boundary_quantiles['Q50']:.3f}, Q80={boundary_quantiles['Q80']:.3f}, Q90={boundary_quantiles['Q90']:.3f}")
            
            # Definir limiares baseados em quantis
            thresholds = {
                'thin_cloud': quantiles[2]['Q75'] if 2 in quantiles else 0.2,
                'shadow': quantiles[3]['Q75'] if 3 in quantiles else 0.15,
                'thick_cloud': quantiles[1]['Q50'] if 1 in quantiles else 0.3,
                'clear': quantiles[0]['Q80'] if 0 in quantiles else 0.7
            }
            
            # Normalizar entropy (z-score) para difficulty score
            entropy_mean = np.mean(entropy_values)
            entropy_std = np.std(entropy_values)
            
            for patch in all_patches_info:
                # Difficulty score baseado apenas em entropia
                entropy_z = (patch['entropy'] - entropy_mean) / (entropy_std + 1e-8)
                patch['difficulty_score'] = entropy_z
            
            print("\nLimiares definidos por quantis:")
            print(f"  Thin Cloud: ≥ {thresholds['thin_cloud']:.2f} (Q75)")
            print(f"  Shadow: ≥ {thresholds['shadow']:.2f} (Q75)")
            print(f"  Thick Cloud: ≥ {thresholds['thick_cloud']:.2f} (Q50)")
            print(f"  Clear: ≥ {thresholds['clear']:.2f} (Q80)")
            
            # Categorizar patches usando os limiares
            candidates = {
                'thin_cloud': [],
                'shadow': [],
                'thick_cloud': [],
                'clear': []
            }
            
            for patch in all_patches_info:
                # Thin cloud
                if 2 in patch['classes'] and patch['fractions'][2] >= thresholds['thin_cloud']:
                    candidates['thin_cloud'].append(patch)
                
                # Shadow
                if 3 in patch['classes'] and patch['fractions'][3] >= thresholds['shadow']:
                    candidates['shadow'].append(patch)
                
                # Thick cloud
                if 1 in patch['classes'] and patch['fractions'][1] >= thresholds['thick_cloud']:
                    candidates['thick_cloud'].append(patch)
                
                # Clear
                if patch['fractions'][0] >= thresholds['clear']:
                    candidates['clear'].append(patch)
            
            print(f"\nCandidatos encontrados:")
            for cat, cands in candidates.items():
                print(f"  {cat}: {len(cands)} patches")
            
            # Salvar informações de seleção para reprodutibilidade
            self.selection_info = {
                'thresholds': thresholds,
                'quantiles': quantiles,
                'boundary_quantiles': boundary_quantiles,
                'normalization_params': {
                    'entropy': {'mean': entropy_mean, 'std': entropy_std}
                }
            }
            
            # Selecionar exemplos: pegar MEDIANO do top-10 por dificuldade
            selected = []
            
            def select_median_from_top(category_name, candidate_list, top_k=10):
                if not candidate_list:
                    return None
                
                # Ordenar por difficulty_score (entropia normalizada)
                sorted_candidates = sorted(candidate_list, 
                                        key=lambda x: x['difficulty_score'], 
                                        reverse=True)
                
                # Pegar top-K
                top_candidates = sorted_candidates[:min(top_k, len(sorted_candidates))]
                
                # Escolher o mediano do top-K
                median_idx = len(top_candidates) // 2
                selected_patch = top_candidates[median_idx]
                
                return (selected_patch['batch_idx'], 
                    selected_patch['image_idx'], 
                    category_name,
                    selected_patch)
            
            # 1. Thin Cloud (mais crítico)
            result = select_median_from_top('Thin Cloud', candidates['thin_cloud'])
            if result:
                selected.append(result)
            
            # 2. Shadow (segundo mais crítico)
            result = select_median_from_top('Shadow', candidates['shadow'])
            if result and (result[0], result[1]) not in [(s[0], s[1]) for s in selected]:
                selected.append(result)
            
            # 3. Thick Cloud
            result = select_median_from_top('Thick Cloud', candidates['thick_cloud'])
            if result and (result[0], result[1]) not in [(s[0], s[1]) for s in selected]:
                selected.append(result)
            
            # 4. Clear (baseline)
            result = select_median_from_top('Clear', candidates['clear'], top_k=5)
            if result and (result[0], result[1]) not in [(s[0], s[1]) for s in selected]:
                selected.append(result)
            
            # Salvar índices e metadados
            self.qualitative_indices = selected[:4]  # 4 exemplos
            
            print("\n" + "="*40)
            print("EXEMPLOS SELECIONADOS (mediano do top-K por entropia):")
            print("="*40)
            for idx, (b, i, cat, info) in enumerate(self.qualitative_indices):
                print(f"  {idx+1}. {cat}:")
                print(f"     Batch {b}, Img {i}")
                print(f"     Entropia: {info['entropy']:.3f}")
                print(f"     Densidade de fronteiras: {info['boundary_density']:.3f}")
                print(f"     Classes: {[CLASS_NAMES[c] for c in info['classes']]}")
                frac_str = ", ".join([f"{CLASS_NAMES[c]}:{info['fractions'][c]:.1%}" 
                                    for c in range(4) if info['fractions'][c] > 0.05])
                print(f"     Frações: {frac_str}")
        
        # Coletar os exemplos para este modelo
        examples = []
        target_n = len(self.qualitative_indices)
        
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_loader):
                # Verificar se este batch tem exemplos selecionados
                indices_in_this_batch = [
                    (b_idx, img_idx, cat, info) 
                    for b_idx, img_idx, cat, info in self.qualitative_indices 
                    if b_idx == batch_idx
                ]
                
                if not indices_in_this_batch:
                    continue
                
                images = images.to(self.device).float()
                masks = masks.to(self.device)
                
                # Fazer predição com modelo único (sem normalização)
                output = model(images)
                pred_probs = F.softmax(output, dim=1)
                pred_masks = pred_probs.argmax(dim=1)
                
                # Coletar os exemplos
                for _, img_idx, category, info in indices_in_this_batch:
                    if img_idx >= images.size(0):
                        continue
                    
                    img = images[img_idx].cpu()
                    true_mask = masks[img_idx].cpu()
                    pred_mask = pred_masks[img_idx].cpu()
                    
                    # Calcular métricas detalhadas
                    accuracy = (pred_mask == true_mask).float().mean().item()
                    
                    # Métricas por classe (IoU, OE, CE)
                    class_metrics = {}
                    for c in range(4):
                        true_c = (true_mask == c)
                        pred_c = (pred_mask == c)
                        
                        if true_c.any() or pred_c.any():
                            # Intersection over Union
                            intersection = (true_c & pred_c).float().sum()
                            union = (true_c | pred_c).float().sum()
                            iou = (intersection / (union + 1e-6)).item()
                            
                            # Omission Error (FN / (TP + FN))
                            fn = (true_c & ~pred_c).float().sum()
                            tp = intersection
                            oe = (fn / (tp + fn + 1e-6)).item() if true_c.any() else 0
                            
                            # Commission Error (FP / (TP + FP))
                            fp = (~true_c & pred_c).float().sum()
                            ce = (fp / (tp + fp + 1e-6)).item() if pred_c.any() else 0
                            
                            class_metrics[c] = {
                                'iou': iou,
                                'oe': oe,
                                'ce': ce,
                                'present_in_gt': true_c.any().item()
                            }
                    
                    example_data = {
                        'image': img.numpy(),
                        'true_mask': true_mask.numpy(),
                        'pred_mask': pred_mask.numpy(),
                        'accuracy': accuracy,
                        'batch_idx': batch_idx,
                        'image_idx': img_idx,
                        'category': category,
                        'class_metrics': class_metrics,
                        'metadata': info
                    }
                    
                    examples.append(example_data)
                
                # Parar quando coletar todos os exemplos necessários
                if len(examples) >= target_n:
                    break
        
        # Salvar nos resultados
        if model_name not in self.manager.results:
            print(f"Aviso: {model_name} não está nos resultados ainda")
            return
        
        if not hasattr(self.manager.results[model_name], 'additional_info'):
            self.manager.results[model_name].additional_info = {}
        
        self.manager.results[model_name].additional_info['qualitative_examples'] = examples
        
        print(f"\n✔ {model_name}: {len(examples)} exemplos coletados")

    def _compute_aggregated_metrics_by_category(self, model_name, models, test_loader, 
                                           use_ensemble, normalize_imgs):
        """
        Calcula métricas IoU agregadas por categoria em TODO o conjunto de teste.
        Executado durante evaluate_model() após _collect_qualitative_examples()
        """
        import torch
        import torch.nn.functional as F
        import numpy as np
        from collections import defaultdict
        
        if not hasattr(self, 'selection_info') or self.selection_info is None:
            print("Pulando métricas agregadas: selection_info não disponível")
            return
        
        print("\n" + "="*60)
        print("CALCULANDO MÉTRICAS AGREGADAS POR CATEGORIA")
        print("="*60)
        
        thresholds = self.selection_info['thresholds']
        
        # Estrutura para armazenar IOUs por categoria
        category_metrics = {
            'Clear': [],
            'Thick Cloud': [],
            'Thin Cloud': [],
            'Shadow': [],
            'Mixed/Transition': []
        }
        
        if normalize_imgs:
            mean, std = get_normalization_stats(self.device, False, SENTINEL_BANDS)
        else:
            mean = std = None
        
        # Processar todo o dataset
        for model in models:
            model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_loader):
                images = images.to(self.device).float()
                masks = masks.to(self.device)
                
                images_in = normalize_images(images, mean, std) if normalize_imgs else images
                
                # Fazer predição
                if use_ensemble:
                    predictions = []
                    for model in models:
                        output = model(images_in)
                        predictions.append(F.softmax(output, dim=1))
                    avg_prediction = torch.stack(predictions).mean(dim=0)
                    pred_masks = avg_prediction.argmax(dim=1)
                else:
                    model = models[0]
                    output = model(images_in)
                    pred_probs = F.softmax(output, dim=1)
                    pred_masks = pred_probs.argmax(dim=1)
                
                # Processar cada imagem no batch
                for img_idx in range(masks.size(0)):
                    true_mask = masks[img_idx].cpu()
                    pred_mask = pred_masks[img_idx].cpu()
                    
                    # Calcular propriedades do patch
                    total_pixels = true_mask.numel()
                    fractions = {}
                    for c in range(4):
                        fractions[c] = (true_mask == c).sum().item() / total_pixels
                    
                    # Calcular boundary density
                    bd_h = (true_mask[:, 1:] != true_mask[:, :-1]).float().sum()
                    bd_v = (true_mask[1:, :] != true_mask[:-1, :]).float().sum()
                    boundary_density = (bd_h + bd_v).item() / (2.0 * true_mask.numel())
                    
                    # Classificar o patch em categorias
                    patch_categories = []
                    
                    # Clear
                    if fractions[0] >= thresholds['clear']:
                        patch_categories.append(('Clear', 0))
                    
                    # Thick Cloud
                    if 1 in true_mask.unique().tolist() and fractions[1] >= thresholds['thick_cloud']:
                        patch_categories.append(('Thick Cloud', 1))
                    
                    # Thin Cloud
                    if 2 in true_mask.unique().tolist() and fractions[2] >= thresholds['thin_cloud']:
                        patch_categories.append(('Thin Cloud', 2))
                    
                    # Shadow
                    if 3 in true_mask.unique().tolist() and fractions[3] >= thresholds['shadow']:
                        patch_categories.append(('Shadow', 3))
                    
                    # Mixed/Transition
                    significant_classes = sum(1 for f in fractions.values() if f > 0.1)
                    if (significant_classes >= thresholds.get('mixed_min_classes', 3) or 
                        boundary_density >= thresholds['mixed_boundary']):
                        patch_categories.append(('Mixed/Transition', None))
                    
                    # Calcular métricas para cada categoria aplicável
                    for category_name, target_class in patch_categories:
                        if target_class is not None:
                            # Calcular IoU para a classe específica
                            true_c = (true_mask == target_class)
                            pred_c = (pred_mask == target_class)
                            
                            intersection = (true_c & pred_c).float().sum()
                            union = (true_c | pred_c).float().sum()
                            iou = (intersection / (union + 1e-6)).item()
                            
                            category_metrics[category_name].append(iou)
                        else:
                            # Para Mixed/Transition, usar accuracy
                            accuracy = (pred_mask == true_mask).float().mean().item()
                            category_metrics[category_name].append(accuracy)
        
        # Calcular estatísticas
        aggregated_results = {}
        for category, values in category_metrics.items():
            if values:
                aggregated_results[category] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values),
                    'values': values  # Guardar para análise posterior se necessário
                }
            else:
                aggregated_results[category] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'count': 0,
                    'values': []
                }
        
        # Salvar nos resultados
        if model_name not in self.manager.results:
            return
        
        if not hasattr(self.manager.results[model_name], 'additional_info'):
            self.manager.results[model_name].additional_info = {}
        
        self.manager.results[model_name].additional_info['aggregated_metrics_by_category'] = aggregated_results
        
        # Imprimir resumo
        print(f"\nMétricas agregadas para {model_name}:")
        print("-"*60)
        for category in ['Clear', 'Thick Cloud', 'Thin Cloud', 'Shadow', 'Mixed/Transition']:
            data = aggregated_results[category]
            if data['count'] > 0:
                print(f"{category:20s}: IoU/Acc={data['mean']:.3f} ± {data['std']:.3f} (n={data['count']})")
            else:
                print(f"{category:20s}: --- (n=0)")

    def _benchmark_inference_costs(
        self,
        model_name,
        models,
        test_loader,
        use_ensemble,
        normalize_imgs,
        warmup_batches: int = 5,
        measure_batches: int | None = None,  # None => usa critério por amostra/tempo
        repetitions: int = 5,
        min_patches: int = 500,
        min_measured_ms: float = 5000.0,     # em ms
    ):
        """
        Micro-benchmark de inferência (forward pass):
        - Latência (ms/patch), vazão (patch/s)
        - Pico de memória de GPU: allocated e reserved (incluindo pesos)

        Observação de escopo: mede *somente* o forward + pré-processamento no _prep_images,
        não inclui criação de patches, merging, IO, ATTA/ensemble fora de get_predictions
        nem RAM do host. No paper, tempo/RAM são do pipeline por cena.  [ver Seção 2.3; Tabela A.4]
        """
        import time
        import torch
        import numpy as np
        import torch.nn.functional as F

        # Flags efetivas (somente CloudS2Mask ensemble normaliza e usa ensemble)
        is_c2m = (model_name == "CloudS2Mask ensemble")
        use_ensemble_eff   = bool(use_ensemble)   if is_c2m else False
        normalize_imgs_eff = bool(normalize_imgs) if is_c2m else False

        if model_name not in self.manager.results:
            print(f"[benchmark] '{model_name}' ainda não está em results; pulando.")
            return

        cuda = torch.cuda.is_available()
        device_name = (torch.cuda.get_device_name(0) if cuda else "CPU")
        # Mantém os modelos no CPU entre repetições para que o pico inclua os pesos
        for m in models:
            m.to("cpu").eval()

        dl_bs = getattr(test_loader, "batch_size", None)
        patch_hw_global = None

        # Acumuladores por repetição
        per_run = []
        lat_list, thr_list = [], []
        alloc_list, reserv_list = [], []

        def _stats(seq: list[float] | None):
            if not seq:
                return None
            arr = np.asarray(seq, dtype=float)
            return {
                "median": float(np.median(arr)),
                "p5":     float(np.percentile(arr, 5)),
                "p95":    float(np.percentile(arr, 95)),
                "mean":   float(arr.mean()),
                "std":    float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            }

        print("\n[Benchmark de inferência — início]")
        print(f"  Dispositivo: {device_name}")
        print(f"  Política de amostragem: warmup={warmup_batches} | "
            f"parada por lotes={measure_batches if measure_batches is not None else '—'} | "
            f"min_patches={min_patches} | min_tempo={min_measured_ms/1000:.1f}s | "
            f"repetições={repetitions}")

        for rep in range(int(repetitions)):
            # ---------- RESET DE MEMÓRIA *ANTES* DE MOVER PESOS ----------
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)

            # ---------- Move pesos para GPU depois do reset ----------
            for m in models:
                m.to(self.device).eval()

            # Normalização (se aplicável)
            if normalize_imgs_eff:
                mean, std = get_normalization_stats(self.device, False, SENTINEL_BANDS)
            else:
                mean = std = None

            # Helpers fechados sobre mean/std
            def _prep_images(imgs: torch.Tensor) -> torch.Tensor:
                # ⚠️ Converte e torna contíguo ANTES de enviar à GPU
                if imgs.dtype != torch.float32:
                    imgs = imgs.float()
                if not imgs.is_contiguous():
                    imgs = imgs.contiguous()

                # Cópia H2D assíncrona (precisa pin_memory=True no DataLoader)
                imgs = imgs.to(self.device, non_blocking=True)

                if normalize_imgs_eff:
                    return normalize_images(imgs, mean, std)
                return imgs

            def _forward(imgs: torch.Tensor):
                imgs_in = _prep_images(imgs)
                _ = get_predictions(
                    models, imgs_in,
                    use_ensemble=use_ensemble_eff,
                    return_probs=False
                )

            # ---------- Laço de medição ----------
            total_ms = 0.0
            total_patches = 0
            measured_ms = 0.0
            measured_patches = 0
            patch_hw_local = None

            with torch.inference_mode():
                for batch_idx, (images, _) in enumerate(test_loader):
                    if patch_hw_local is None:
                        patch_hw_local = tuple(images.shape[2:])

                    if batch_idx < warmup_batches:
                        _forward(images)
                        continue

                    if cuda:
                        start = torch.cuda.Event(enable_timing=True)
                        end   = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize(self.device)
                        start.record()
                        _forward(images)
                        end.record()
                        torch.cuda.synchronize(self.device)
                        elapsed_ms = start.elapsed_time(end)
                    else:
                        t0 = time.perf_counter()
                        _forward(images)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    bsz = int(images.shape[0])
                    total_ms      += elapsed_ms
                    total_patches += bsz
                    measured_ms   += elapsed_ms
                    measured_patches += bsz

                    # Critério de parada
                    if measure_batches is not None:
                        if (batch_idx - warmup_batches + 1) >= measure_batches:
                            break
                    else:
                        if (measured_patches >= min_patches) or (measured_ms >= min_measured_ms):
                            break

            if patch_hw_global is None:
                patch_hw_global = patch_hw_local

            if total_patches == 0:
                print(f"  [rep {rep+1}] nenhum batch medido; verifique o test_loader.")
                # Volta modelos para CPU e segue
                for m in models:
                    m.to("cpu")
                continue

            # Métricas desta repetição
            latency_ms = total_ms / total_patches
            throughput = (1000.0 * total_patches) / total_ms

            if cuda:
                peak_alloc_mb   = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
                peak_reserved_mb= torch.cuda.max_memory_reserved(self.device)  / (1024.0 ** 2)
            else:
                peak_alloc_mb = peak_reserved_mb = None

            per_run.append({
                "latency_ms_per_patch": float(latency_ms),
                "throughput_patches_per_s": float(throughput),
                "peak_mem_alloc_mb": float(peak_alloc_mb) if peak_alloc_mb is not None else None,
                "peak_mem_reserved_mb": float(peak_reserved_mb) if peak_reserved_mb is not None else None,
                "measured_patches": int(total_patches),
                "measured_ms": float(total_ms),
            })

            lat_list.append(latency_ms)
            thr_list.append(throughput)
            if peak_alloc_mb is not None:
                alloc_list.append(peak_alloc_mb)
                reserv_list.append(peak_reserved_mb)

            # Limpa para próxima repetição
            for m in models:
                m.to("cpu")

        # ---------- Agregados ----------
        summary = {
            "latency_ms_per_patch": _stats(lat_list),
            "throughput_patches_per_s": _stats(thr_list),
            "peak_mem_alloc_mb": _stats(alloc_list) if alloc_list else None,
            "peak_mem_reserved_mb": _stats(reserv_list) if reserv_list else None,
        }

        # Persistência
        info = getattr(self.manager.results[model_name], "additional_info", {}) or {}
        info.setdefault("compute_cost", {})
        cc = info["compute_cost"]

        cc.update({
            "device_name": device_name,
            "policy": {
                "measurement_scope": (
                    "Forward pass na GPU (inclui cópia H2D em _prep_images); "
                    "não inclui criação de patches, merging, IO; "
                    "RAM pico reportada é de GPU, não RAM do host."
                ),
                "warmup_batches": int(warmup_batches),
                "measure_batches": (None if measure_batches is None else int(measure_batches)),
                "repetitions": int(repetitions),
                "min_patches": int(min_patches),
                "min_measured_ms": float(min_measured_ms),
                "batch_size_measured": (int(dl_bs) if dl_bs is not None else None),
                "patch_shape": patch_hw_global,
                "flags_effective": {
                    "is_clouds2mask_ensemble": is_c2m,
                    "use_ensemble": use_ensemble_eff,
                    "normalize_imgs": normalize_imgs_eff,
                },
            },
            "per_run": per_run,
            "summary": summary,
            "timestamp_benchmark": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        })

        # Campos legados (para gráficos já existentes) — usamos as MEDIANAS
        if summary["latency_ms_per_patch"]:
            cc["latency_ms_per_patch"] = summary["latency_ms_per_patch"]["median"]
        if summary["throughput_patches_per_s"]:
            cc["throughput_patches_per_s"] = summary["throughput_patches_per_s"]["median"]
        # Para compatibilidade com 'peak_mem_mb' antigo, usar 'reserved' (mais estável)
        if cuda and summary["peak_mem_reserved_mb"]:
            cc["peak_mem_mb"] = summary["peak_mem_reserved_mb"]["median"]
        else:
            cc["peak_mem_mb"] = None

        self.manager.results[model_name].additional_info = info

        # ---------- Impressão-resumo ----------
        print("\n[Benchmark de inferência — resumo]")
        if summary["latency_ms_per_patch"]:
            s = summary["latency_ms_per_patch"]
            print(f"  Latência (ms/patch): mediana {s['median']:.2f}  [p5 {s['p5']:.2f} ; p95 {s['p95']:.2f}]")
        if summary["throughput_patches_per_s"]:
            s = summary["throughput_patches_per_s"]
            print(f"  Throughput (patch/s): mediana {s['median']:.2f}  [p5 {s['p5']:.2f} ; p95 {s['p95']:.2f}]")
        if cuda and summary["peak_mem_alloc_mb"] and summary["peak_mem_reserved_mb"]:
            sa = summary["peak_mem_alloc_mb"]; sr = summary["peak_mem_reserved_mb"]
            print(f"  Memória pico GPU (allocated/reserved, MB): "
                f"{sa['median']:.0f} / {sr['median']:.0f}  "
                f"[allocated p95 {sa['p95']:.0f}; reserved p95 {sr['p95']:.0f}]")
        print("  Escopo: forward pass de GPU; paper reporta RAM do host e tempo por cena (E2E).")