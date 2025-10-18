from .metrics import CLASS_NAMES


class EvaluationPrinter:
    """Gerencia impressão de resultados de avaliação"""
    
    def __init__(self, manager):
        self.manager = manager
    
    def print_cached_summary(self, model_name):
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
        
        # Resumo dos limiares
        print("\n" + "="*60)
        print("RESUMO DOS LIMIARES (val-cal → aplicados no TESTE)")
        print("="*60)

        aux = getattr(result, "additional_info", {}) or {}

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
    
    def print_metrics(self, metrics):
        """Imprime métricas formatadas"""
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
    
    def print_summary(self, model_name):
        """Imprime resumo geral"""
        print("\n" + "="*60)
        print("RESUMO DOS LIMIARES (val-cal → aplicados no TESTE)")
        print("="*60)
        
        model_result = self.manager.results[model_name]
        aux = getattr(model_result, "additional_info", {}) or {}

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
    
    def print_threshold_summary(self, model_name, experiments, result_obj, df_thresh_valcal):
        """Imprime resumo da otimização de limiares"""
        print("\n========== RESUMO t* (VAL → TESTE) ==========")
        for exp in experiments:
            base_boa = result_obj.boa_baseline.get(exp, float('nan'))
            try:
                boa_aplicada = float(
                    df_thresh_valcal.loc[df_thresh_valcal['Experiment'] == exp, 'Median BOA'].iloc[0]
                )
            except Exception:
                boa_aplicada = float('nan')

            bv = result_obj.additional_info['thr_results_valcal'][exp]['best_threshold']
            print(f"\n[{exp}]")
            print(f"  BOA (argmax, TESTE): {base_boa:.4f}")
            print(f"  t* (calibrado na VAL): {bv:.2f} | BOA aplicada (TESTE): {boa_aplicada:.4f} | Δ: {(boa_aplicada - base_boa):+.4f}")