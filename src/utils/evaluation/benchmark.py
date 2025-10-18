import time
import torch
import numpy as np

from .metrics import (
    get_normalization_stats,
    get_predictions,
    normalize_images,
    SENTINEL_BANDS
)


class InferenceBenchmark:
    """Gerencia benchmarks de inferência"""
    
    def __init__(self, device, manager):
        self.device = device
        self.manager = manager
    
    def benchmark_inference_costs(
        self,
        model_name,
        models,
        test_loader,
        use_ensemble,
        normalize_imgs,
        warmup_batches: int = 5,
        measure_batches: int | None = None,
        repetitions: int = 5,
        min_patches: int = 500,
        min_measured_ms: float = 5000.0,
    ):
        """
        Micro-benchmark de inferência (forward pass)
        """
        # Flags efetivas
        is_c2m = (model_name == "CloudS2Mask ensemble")
        use_ensemble_eff = bool(use_ensemble) if is_c2m else False
        normalize_imgs_eff = bool(normalize_imgs) if is_c2m else False

        if model_name not in self.manager.results:
            print(f"[benchmark] '{model_name}' ainda não está em results; pulando.")
            return

        cuda = torch.cuda.is_available()
        device_name = (torch.cuda.get_device_name(0) if cuda else "CPU")
        
        # Mantém os modelos no CPU entre repetições
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
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            }

        print("\n[Benchmark de inferência — início]")
        print(f"  Dispositivo: {device_name}")
        print(f"  Política de amostragem: warmup={warmup_batches} | "
              f"parada por lotes={measure_batches if measure_batches is not None else '—'} | "
              f"min_patches={min_patches} | min_tempo={min_measured_ms/1000:.1f}s | "
              f"repetições={repetitions}")

        for rep in range(int(repetitions)):
            # Reset de memória antes de mover pesos
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)

            # Move pesos para GPU
            for m in models:
                m.to(self.device).eval()

            # Normalização
            if normalize_imgs_eff:
                mean, std = get_normalization_stats(self.device, False, SENTINEL_BANDS)
            else:
                mean = std = None

            # Helpers
            def _prep_images(imgs: torch.Tensor) -> torch.Tensor:
                if imgs.dtype != torch.float32:
                    imgs = imgs.float()
                if not imgs.is_contiguous():
                    imgs = imgs.contiguous()

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

            # Laço de medição
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
                        end = torch.cuda.Event(enable_timing=True)
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
                    total_ms += elapsed_ms
                    total_patches += bsz
                    measured_ms += elapsed_ms
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
                for m in models:
                    m.to("cpu")
                continue

            # Métricas desta repetição
            latency_ms = total_ms / total_patches
            throughput = (1000.0 * total_patches) / total_ms

            if cuda:
                peak_alloc_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
                peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024.0 ** 2)
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

        # Agregados
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

        # Campos legados
        if summary["latency_ms_per_patch"]:
            cc["latency_ms_per_patch"] = summary["latency_ms_per_patch"]["median"]
        if summary["throughput_patches_per_s"]:
            cc["throughput_patches_per_s"] = summary["throughput_patches_per_s"]["median"]
        if cuda and summary["peak_mem_reserved_mb"]:
            cc["peak_mem_mb"] = summary["peak_mem_reserved_mb"]["median"]
        else:
            cc["peak_mem_mb"] = None

        self.manager.results[model_name].additional_info = info

        # Impressão-resumo
        print("\n[Benchmark de inferência — resumo]")
        if summary["latency_ms_per_patch"]:
            s = summary["latency_ms_per_patch"]
            print(f"  Latência (ms/patch): mediana {s['median']:.2f}  [p5 {s['p5']:.2f} ; p95 {s['p95']:.2f}]")
        if summary["throughput_patches_per_s"]:
            s = summary["throughput_patches_per_s"]
            print(f"  Throughput (patch/s): mediana {s['median']:.2f}  [p5 {s['p5']:.2f} ; p95 {s['p95']:.2f}]")
        if cuda and summary["peak_mem_alloc_mb"] and summary["peak_mem_reserved_mb"]:
            sa = summary["peak_mem_alloc_mb"]
            sr = summary["peak_mem_reserved_mb"]
            print(f"  Memória pico GPU (allocated/reserved, MB): "
                  f"{sa['median']:.0f} / {sr['median']:.0f}  "
                  f"[allocated p95 {sa['p95']:.0f}; reserved p95 {sr['p95']:.0f}]")
        print("  Escopo: forward pass de GPU; paper reporta RAM do host e tempo por cena (E2E).")