"""Model prediction utilities including ensemble inference."""

from typing import List, Union

import torch


def ensemble_inference(
    models: List[torch.nn.Module],
    images: torch.Tensor,
    return_probs: bool = False,
) -> torch.Tensor:
    """Average softmax probabilities across multiple models.

    Args:
        models: List of models in eval mode.
        images: Input batch (B, C, H, W).
        return_probs: If True, returns averaged probabilities (B, K, H, W).
            If False, returns argmax class indices (B, H, W).

    Returns:
        Averaged probabilities or predicted class indices.
    """
    with torch.no_grad():
        probs = torch.stack(
            [torch.softmax(m(images), dim=1) for m in models]
        ).mean(dim=0)

    if return_probs:
        return probs
    return torch.argmax(probs, dim=1)


def get_predictions(
    models: Union[torch.nn.Module, List[torch.nn.Module]],
    images: torch.Tensor,
    use_ensemble: bool = True,
    return_probs: bool = False,
) -> torch.Tensor:
    """Get predictions from a single model or ensemble.

    Args:
        models: Single model or list of models.
        images: Input batch (B, C, H, W).
        use_ensemble: If True and multiple models given, uses ensemble.
        return_probs: If True, returns probabilities instead of class indices.

    Returns:
        Predictions tensor.
    """
    if not isinstance(models, list):
        models = [models]

    if use_ensemble and len(models) > 1:
        return ensemble_inference(models, images, return_probs=return_probs)

    with torch.no_grad():
        output = models[0](images)
        probs = torch.softmax(output, dim=1)

    if return_probs:
        return probs
    return torch.argmax(probs, dim=1)


def load_models(
    model_paths: List[str],
    device: Union[str, torch.device],
) -> List[torch.nn.Module]:
    """Load saved PyTorch models and set to eval mode.

    Args:
        model_paths: Paths to saved model files.
        device: Device to load models onto.

    Returns:
        List of loaded models in evaluation mode.
    """
    models = []
    for path in model_paths:
        print(f"Loading model: {path}")
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()
        models.append(model)
    return models