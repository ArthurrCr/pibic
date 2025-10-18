import torch
from pathlib import Path

from .metrics import (
    evaluate_clouds2mask,
    compute_metrics,
    plot_confusion_matrix,
    CLASS_NAMES
)
from .boa_metrics import (
    evaluate_test_dataset,
    evaluate_test_dataset_with_thresholds,
    find_optimal_threshold_by_patch
)
from .cache_manager import CacheManager
from .qualitative_analysis import QualitativeAnalyzer
from .benchmark import InferenceBenchmark
from .evaluation_helpers import EvaluationPrinter


class ModelEvaluator:
    """Coordena a avaliação completa de modelos de segmentação"""
    
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
        
        # Componentes auxiliares
        self.cache_manager = CacheManager(self.cache_dir, self.manager)
        self.qualitative_analyzer = QualitativeAnalyzer(self.device, self.manager)
        self.benchmark = InferenceBenchmark(self.device, self.manager)
        self.printer = EvaluationPrinter(self.manager)
        
        # Carregar cache existente
        self.cache_manager.auto_load_existing_results()
    
    def evaluate_model(self, model_name, models, test_loader, val_loader=None, 
                      use_ensemble=None, normalize_imgs=None):
        """Avaliação completa de um modelo"""
        
        # Verificar cache
        if model_name in self.manager.results:
            print(f"\n{'='*60}")
            print(f"RESULTADOS JÁ EXISTEM PARA: {model_name}")
            print(f"{'='*60}")
            print("✔ Usando resultados do cache")
            self.printer.print_cached_summary(model_name)
            self._plot_all_results(model_name)
            return
        
        # Configurar avaliação
        use_ensemble, normalize_imgs = self._setup_config(
            model_name, use_ensemble, normalize_imgs
        )
        
        print(f"\n{'='*60}")
        print(f"INICIANDO AVALIAÇÃO: {model_name}")
        print(f"{'='*60}")
        print(f"Número de modelos: {len(models)}")
        print(f"Usando ensemble: {use_ensemble}")
        print(f"Normalizar imagens: {normalize_imgs}")
        
        # 1. Matriz de Confusão
        self._evaluate_confusion_matrix(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )
        
        # 2. BOA baseline
        self._evaluate_patches(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )
        
        # 3. Análise qualitativa
        self.qualitative_analyzer.collect_qualitative_examples(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )
        
        # 4. Métricas agregadas por categoria
        self.qualitative_analyzer.compute_aggregated_metrics_by_category(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )
        
        # 5. Otimização de limiares
        if val_loader is None:
            raise ValueError("val_loader é obrigatório para calibração")
        
        self._calculate_optimal_thresholds(
            model_name, models, test_loader, use_ensemble, normalize_imgs, val_loader
        )
        
        # 6. Número de parâmetros
        self._save_model_params(model_name, models)
        
        # 7. Benchmark
        self.benchmark.benchmark_inference_costs(
            model_name, models, test_loader, use_ensemble, normalize_imgs
        )
        
        # 8. Resumo e visualizações
        self.printer.print_summary(model_name)
        self._plot_all_results(model_name)
        
        # 9. Salvar cache
        self.cache_manager.save_complete_state(model_name)
    
    def _setup_config(self, model_name, use_ensemble, normalize_imgs):
        """Configura parâmetros de avaliação"""
        if model_name in self.model_configs:
            config = self.model_configs[model_name]
            use_ensemble = config["use_ensemble"] if use_ensemble is None else use_ensemble
            normalize_imgs = config["normalize_imgs"] if normalize_imgs is None else normalize_imgs
        else:
            use_ensemble = use_ensemble if use_ensemble is not None else False
            normalize_imgs = normalize_imgs if normalize_imgs is not None else False
        return use_ensemble, normalize_imgs
    
    def _evaluate_confusion_matrix(self, model_name, models, test_loader, 
                                  use_ensemble, normalize_imgs):
        """Avalia matriz de confusão"""
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
        self.printer.print_metrics(metrics)
    
    def _evaluate_patches(self, model_name, models, test_loader, 
                         use_ensemble, normalize_imgs):
        """Avalia BOA por patch"""
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
                                     use_ensemble, normalize_imgs, val_loader):
        """Calibra limiares ótimos"""
        print("\n" + "="*60)
        print("CALCULANDO LIMIAR ÓTIMO (t*) POR PATCH")
        print("="*60)
        
        experiments = ['cloud/no cloud', 'cloud shadow', 'valid/invalid']
        t_star_valcal = {}
        thr_results_valcal = {}
        
        # Calibrar na validação
        print("\n--- Calibração de t* na VALIDAÇÃO ---")
        for experiment in experiments:
            print(f"  · t* (val-cal) para {experiment}...")
            res_val = find_optimal_threshold_by_patch(
                test_loader=val_loader,
                models=models,
                experiment=experiment,
                device=self.device,
                use_ensemble=use_ensemble,
                normalize_imgs=normalize_imgs
            )
            res_val['calibrated_on'] = 'val'
            thr_results_valcal[experiment] = res_val
            t_star_valcal[experiment] = res_val['best_threshold']
        
        # Aplicar no teste
        df_thresh_valcal = evaluate_test_dataset_with_thresholds(
            test_loader,
            models,
            t_star_valcal,
            device=str(self.device),
            use_ensemble=use_ensemble,
            normalize_imgs=normalize_imgs
        )
        
        # Salvar resultados
        for experiment in experiments:
            self.manager.save_boa_results(
                model_name,
                threshold_results=thr_results_valcal[experiment],
                experiment=experiment
            )
        
        # Guardar informações adicionais
        result_obj = self.manager.results[model_name]
        if not hasattr(result_obj, 'additional_info'):
            result_obj.additional_info = {}
        
        result_obj.additional_info.update({
            'thresholds_source_canonical': 'val',
            't_star_valcal': t_star_valcal,
            'thr_results_valcal': thr_results_valcal,
            'df_thresh_valcal': df_thresh_valcal,
            't_star': t_star_valcal,
            'df_thresh': df_thresh_valcal
        })
        
        # Imprimir resumo
        self.printer.print_threshold_summary(model_name, experiments, result_obj, df_thresh_valcal)
    
    def _save_model_params(self, model_name, models):
        """Salva número de parâmetros"""
        n_params = sum(p.numel() for m in models for p in m.parameters())
        if not hasattr(self.manager.results[model_name], 'additional_info'):
            self.manager.results[model_name].additional_info = {}
        self.manager.results[model_name].additional_info['n_parameters'] = n_params
    
    def _plot_all_results(self, model_name):
        """Plota resultados"""
        plot_confusion_matrix(
            self.manager.results[model_name].confusion_matrix, 
            normalize=True,
            title=f'Matriz de Confusão - {model_name}'
        )