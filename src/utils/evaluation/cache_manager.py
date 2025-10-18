import pickle
import hashlib
from pathlib import Path


class CacheManager:
    """Gerencia cache de resultados de avaliação"""
    
    def __init__(self, cache_dir, manager):
        self.cache_dir = Path(cache_dir)
        self.manager = manager
        self._model_index = {}
    
    def auto_load_existing_results(self):
        """Carrega automaticamente todos os resultados existentes"""
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
                    print(f"  ✔ Carregado: {model_name}")
                    
            except Exception as e:
                print(f"  ✗ Erro ao carregar {file.name}: {e}")
        
        if loaded_count > 0:
            print(f"\n{loaded_count} avaliação(ões) carregada(s) do cache.")
        else:
            print("Nenhuma avaliação anterior encontrada.")
        
        self._model_index = model_index
    
    def get_cache_path(self, model_name, eval_type):
        """Gera caminho único para cache"""
        name_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        safe_name = model_name.replace("/", "-").replace(" ", "_")
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        filename = f"{safe_name}_{name_hash}_{eval_type}.pkl"
        return self.cache_dir / filename
    
    def save_complete_state(self, model_name):
        """Salva estado completo"""
        if model_name in self.manager.results:
            cache_path = self.get_cache_path(model_name, "evaluation_results")
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
            print(f"\n✔ Métricas de avaliação salvas para: {model_name}")