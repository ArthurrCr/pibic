import torch
import torch.nn.functional as F
import numpy as np

from .metrics import (
    get_normalization_stats,
    normalize_images,
    SENTINEL_BANDS,
    CLASS_NAMES
)


class QualitativeAnalyzer:
    """Gerencia análise qualitativa de modelos"""
    
    def __init__(self, device, manager):
        self.device = device
        self.manager = manager
        self.qualitative_indices = None
        self.selection_info = None
    
    def collect_qualitative_examples(self, model_name, models, test_loader, 
                                   use_ensemble, normalize_imgs, n_examples=5):
        """
        Coleta estratégica de exemplos usando critérios baseados em quantis do dataset
        """
        # Forçar flags corretas para CloudS2Mask ensemble
        is_c2m = (model_name == "CloudS2Mask ensemble")
        use_ensemble = True if is_c2m else False
        normalize_imgs = True if is_c2m else False
        
        if normalize_imgs:
            mean, std = get_normalization_stats(self.device, False, SENTINEL_BANDS)
        else:
            mean = std = None
        
        print("\n" + "="*60)
        print("COLETANDO EXEMPLOS QUALITATIVOS ESTRATÉGICOS")
        print("="*60)
        
        if not hasattr(self, 'qualitative_indices') or self.qualitative_indices is None:
            print("Analisando TODO o conjunto de teste para seleção baseada em quantis...")
            
            all_patches_info = []
            
            for model in models:
                model.eval()
            
            with torch.no_grad():
                for batch_idx, (images, masks) in enumerate(test_loader):
                    images = images.to(self.device).float()
                    masks_cpu = masks.cpu()
                    
                    images_in = normalize_images(images, mean, std) if normalize_imgs else images
                    
                    if use_ensemble:
                        predictions = []
                        for model in models:
                            output = model(images_in)
                            predictions.append(F.softmax(output, dim=1))
                        preds_stack = torch.stack(predictions, dim=0)
                        avg_prediction = preds_stack.mean(dim=0)
                        model_std = preds_stack.std(dim=0)
                        discordance = model_std.mean(dim=1)
                    else:
                        model = models[0]
                        output = model(images_in)
                        avg_prediction = F.softmax(output, dim=1)
                        discordance = torch.zeros(images.size(0), *output.shape[2:]).to(self.device)
                    
                    entropy = -(avg_prediction * torch.log(avg_prediction.clamp_min(1e-8))).sum(dim=1)
                    
                    for img_idx in range(masks_cpu.size(0)):
                        mask = masks_cpu[img_idx]
                        unique_classes = mask.unique().tolist()
                        
                        total_pixels = mask.numel()
                        fractions = {}
                        for c in range(4):
                            fractions[c] = (mask == c).sum().item() / total_pixels
                        
                        bd_h = (mask[:, 1:] != mask[:, :-1]).float().sum()
                        bd_v = (mask[1:, :] != mask[:-1, :]).float().sum()
                        boundary_density = (bd_h + bd_v).item() / (2.0 * mask.numel())
                        
                        patch_info = {
                            'batch_idx': batch_idx,
                            'image_idx': img_idx,
                            'classes': unique_classes,
                            'fractions': fractions,
                            'entropy': entropy[img_idx].mean().cpu().item(),
                            'discordance': discordance[img_idx].mean().cpu().item(),
                            'boundary_density': boundary_density,
                            'n_classes': len(unique_classes)
                        }
                        all_patches_info.append(patch_info)
            
            print(f"Total de patches analisados: {len(all_patches_info)}")
            
            # Calcular quantis
            self._calculate_quantiles(all_patches_info)
            
            # Selecionar exemplos
            self._select_examples(all_patches_info)
        
        # Coletar exemplos para este modelo
        self._collect_model_examples(model_name, models, test_loader, 
                                   use_ensemble, normalize_imgs)
        
    def _calculate_quantiles(self, all_patches_info):
        """Calcula quantis das distribuições"""
        fractions_by_class = {c: [] for c in range(4)}
        boundary_densities = []
        entropy_values = []
        discordance_values = []
        
        for patch in all_patches_info:
            for c in range(4):
                if patch['fractions'][c] > 0:
                    fractions_by_class[c].append(patch['fractions'][c])
            boundary_densities.append(patch['boundary_density'])
            entropy_values.append(patch['entropy'])
            discordance_values.append(patch['discordance'])
        
        quantiles = {}
        for c in range(4):
            if fractions_by_class[c]:
                quantiles[c] = {
                    'Q50': np.percentile(fractions_by_class[c], 50),
                    'Q75': np.percentile(fractions_by_class[c], 75),
                    'Q80': np.percentile(fractions_by_class[c], 80),
                    'Q90': np.percentile(fractions_by_class[c], 90)
                }
        
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
            'clear': quantiles[0]['Q80'] if 0 in quantiles else 0.7,
            'mixed_boundary': boundary_quantiles['Q80'],
            'mixed_min_classes': 3
        }
        
        # Normalizar para difficulty score
        entropy_mean = np.mean(entropy_values)
        entropy_std = np.std(entropy_values)
        discordance_mean = np.mean(discordance_values)
        discordance_std = np.std(discordance_values)
        
        # Salvar informações
        self.selection_info = {
            'thresholds': thresholds,
            'quantiles': quantiles,
            'boundary_quantiles': boundary_quantiles,
            'normalization_params': {
                'entropy': {'mean': entropy_mean, 'std': entropy_std},
                'discordance': {'mean': discordance_mean, 'std': discordance_std}
            }
        }
        
        return all_patches_info, thresholds, entropy_mean, entropy_std, discordance_mean, discordance_std
    
    def _select_examples(self, all_patches_info):
        """Seleciona exemplos baseado em critérios"""
        thresholds = self.selection_info['thresholds']
        norm_params = self.selection_info['normalization_params']
        
        # Calcular difficulty score normalizado
        for patch in all_patches_info:
            entropy_z = (patch['entropy'] - norm_params['entropy']['mean']) / (norm_params['entropy']['std'] + 1e-8)
            discordance_z = (patch['discordance'] - norm_params['discordance']['mean']) / (norm_params['discordance']['std'] + 1e-8)
            patch['difficulty_score'] = entropy_z + discordance_z
        
        print("\nLimiares definidos por quantis:")
        print(f"  Thin Cloud: ≥ {thresholds['thin_cloud']:.2f} (Q75)")
        print(f"  Shadow: ≥ {thresholds['shadow']:.2f} (Q75)")
        print(f"  Thick Cloud: ≥ {thresholds['thick_cloud']:.2f} (Q50)")
        print(f"  Clear: ≥ {thresholds['clear']:.2f} (Q80)")
        print(f"  Mixed (boundary): ≥ {thresholds['mixed_boundary']:.3f} (Q80)")
        
        # Categorizar patches
        candidates = {
            'thin_cloud': [],
            'shadow': [],
            'thick_cloud': [],
            'clear': [],
            'mixed': []
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
            
            # Mixed
            significant_classes = sum(1 for f in patch['fractions'].values() if f > 0.1)
            if (significant_classes >= thresholds['mixed_min_classes'] or 
                patch['boundary_density'] >= thresholds['mixed_boundary']):
                candidates['mixed'].append(patch)
        
        print(f"\nCandidatos encontrados:")
        for cat, cands in candidates.items():
            print(f"  {cat}: {len(cands)} patches")
        
        # Selecionar exemplos
        selected = []
        
        def select_median_from_top(category_name, candidate_list, top_k=10):
            if not candidate_list:
                return None
            
            sorted_candidates = sorted(candidate_list, 
                                     key=lambda x: x['difficulty_score'], 
                                     reverse=True)
            
            top_candidates = sorted_candidates[:min(top_k, len(sorted_candidates))]
            median_idx = len(top_candidates) // 2
            selected_patch = top_candidates[median_idx]
            
            return (selected_patch['batch_idx'], 
                   selected_patch['image_idx'], 
                   category_name,
                   selected_patch)
        
        # Selecionar um de cada categoria
        for cat_name, cat_key in [
            ('Thin Cloud', 'thin_cloud'),
            ('Shadow', 'shadow'),
            ('Mixed/Transition', 'mixed'),
            ('Thick Cloud', 'thick_cloud'),
            ('Clear', 'clear')
        ]:
            result = select_median_from_top(cat_name, candidates[cat_key])
            if result and (result[0], result[1]) not in [(s[0], s[1]) for s in selected]:
                selected.append(result)
        
        self.qualitative_indices = selected[:5]
        
        print("\n" + "="*40)
        print("EXEMPLOS SELECIONADOS (mediano do top-K por dificuldade):")
        print("="*40)
        for idx, (b, i, cat, info) in enumerate(self.qualitative_indices):
            print(f"  {idx+1}. {cat}:")
            print(f"     Batch {b}, Img {i}")
            print(f"     Entropia: {info['entropy']:.3f}, Discordância: {info['discordance']:.3f}")
            print(f"     Densidade de fronteiras: {info['boundary_density']:.3f}")
            print(f"     Classes: {[CLASS_NAMES[c] for c in info['classes']]}")
            frac_str = ", ".join([f"{CLASS_NAMES[c]}:{info['fractions'][c]:.1%}" 
                                for c in range(4) if info['fractions'][c] > 0.05])
            print(f"     Frações: {frac_str}")
    
    def _collect_model_examples(self, model_name, models, test_loader, 
                              use_ensemble, normalize_imgs):
        """Coleta exemplos para um modelo específico"""
        examples = []
        target_n = len(self.qualitative_indices)
        
        if normalize_imgs:
            mean, std = get_normalization_stats(self.device, False, SENTINEL_BANDS)
        else:
            mean = std = None
        
        for model in models:
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
                            intersection = (true_c & pred_c).float().sum()
                            union = (true_c | pred_c).float().sum()
                            iou = (intersection / (union + 1e-6)).item()
                            
                            fn = (true_c & ~pred_c).float().sum()
                            tp = intersection
                            oe = (fn / (tp + fn + 1e-6)).item() if true_c.any() else 0
                            
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
    
    def compute_aggregated_metrics_by_category(self, model_name, models, test_loader, 
                                             use_ensemble, normalize_imgs):
        """
        Calcula métricas IoU agregadas por categoria em TODO o conjunto de teste
        """
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
                    'values': values
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