import torch
import numpy as np
import matplotlib.pyplot as plt
import inspect


def _prepare_video(video, device="cuda:0"):
    video = video.to(device)
    if video.dim() == 2:   # [T, C] -> [1, T, C]
        video = video.unsqueeze(0)
    return video


def _get_ranked_concepts(contribs, mode="positive"):
    """
    Returns ranked concept indices.

    mode:
        - 'positive': rank by largest positive contribution
        - 'absolute': rank by largest absolute contribution
    """
    if mode == "absolute":
        ranked = contribs.abs().argsort(descending=True)
    elif mode == "positive":
        ranked = contribs.argsort(descending=True)
    else:
        raise ValueError("mode must be 'positive' or 'absolute'")
    return ranked.detach().cpu().numpy()


def _get_ranked_local_concepts(contribs_per_time, mode="positive"):
    """
    Returns ranked local concept coordinates as (window_ids, channel_ids).

    contribs_per_time:
        Tensor shaped [T, C] with per-window concept contributions.
    """
    if mode == "absolute":
        ranked_flat = contribs_per_time.abs().reshape(-1).argsort(descending=True)
    elif mode == "positive":
        ranked_flat = contribs_per_time.reshape(-1).argsort(descending=True)
    else:
        raise ValueError("mode must be 'positive' or 'absolute'")

    ranked_flat = ranked_flat.detach().cpu().numpy()
    num_windows, num_concepts = contribs_per_time.shape
    window_ids, channel_ids = np.unravel_index(ranked_flat, (num_windows, num_concepts))
    return window_ids, channel_ids


def compute_original_outputs(cbm_model, explain_instance_fn, device="cuda:0"):
    model = cbm_model.model
    model.eval()

    explanations = []
    original_preds = []
    original_true_probs = []
    original_predclass_probs = []
    original_correct = 0

    with torch.no_grad():
        for i in range(len(cbm_model.X_test)):
            video = _prepare_video(cbm_model.X_test[i], device=device)
            true_idx = int(cbm_model.y_test[i])

            logits, _, _, _ = model(video)   # untouched forward
            probs = torch.softmax(logits, dim=1)

            pred_idx = int(probs.argmax(dim=1).item())

            original_preds.append(pred_idx)
            original_true_probs.append(float(probs[0, true_idx].item()))
            original_predclass_probs.append(float(probs[0, pred_idx].item()))
            original_correct += int(pred_idx == true_idx)

            res = explain_instance_fn(model, video)
            explanations.append(res)

    return {
        "explanations": explanations,
        "original_preds": original_preds,
        "original_true_probs": original_true_probs,
        "original_predclass_probs": original_predclass_probs,
        "original_acc": original_correct / len(cbm_model.X_test),
    }


def _infer_num_concepts(cbm_model):
    x0 = cbm_model.X_test[0]
    if x0.dim() == 2:   # [T, C]
        return x0.shape[1]
    elif x0.dim() == 3: # [1, T, C] or [B, T, C]
        return x0.shape[-1]
    raise ValueError(f"Unexpected input shape: {x0.shape}")


def _compute_metrics_for_intervention(
    cbm_model,
    original_preds,
    original_true_probs,
    original_predclass_probs,
    intervention_selector_fn,
    device="cuda:0",
):
    model = cbm_model.model
    forward_params = inspect.signature(model.forward).parameters
    supports_noisy_channel = "noisy_channel" in forward_params
    supports_attn_intervention = (
        "attn_channel_ids" in forward_params and "attn_mode" in forward_params
    )

    same_pred = 0
    correct = 0
    delta_true = 0.0
    delta_predclass = 0.0
    fraction_kept = 0.0

    with torch.no_grad():
        for i in range(len(cbm_model.X_test)):
            video = _prepare_video(cbm_model.X_test[i], device=device)
            true_idx = int(cbm_model.y_test[i])

            selected = intervention_selector_fn(i)
            if isinstance(selected, dict):
                sample_fraction_kept = selected["fraction_kept"]
                model_kwargs = {k: v for k, v in selected.items() if k != "fraction_kept"}
            else:
                if len(selected) == 4:
                    channel_ids, window_ids, sample_fraction_kept, noisy_channel = selected
                else:
                    channel_ids, window_ids, sample_fraction_kept = selected
                    noisy_channel = False

                model_kwargs = {
                    "channel_ids": channel_ids,
                    "window_ids": window_ids,
                }
                if noisy_channel:
                    model_kwargs["noisy_channel"] = True

            if model_kwargs.get("noisy_channel", False) and not supports_noisy_channel:
                raise ValueError("Model does not support noisy_channel interventions.")
            if (
                ("attn_channel_ids" in model_kwargs or "attn_mode" in model_kwargs)
                and not supports_attn_intervention
            ):
                raise ValueError("Model does not support attention interventions.")

            logits_int, _, _, _ = model(video, **model_kwargs)
            probs_int = torch.softmax(logits_int, dim=1)

            pred_int = int(probs_int.argmax(dim=1).item())
            orig_pred = original_preds[i]

            same_pred += int(pred_int == orig_pred)
            correct += int(pred_int == true_idx)
            delta_true += original_true_probs[i] - float(probs_int[0, true_idx].item())
            delta_predclass += (
                original_predclass_probs[i] - float(probs_int[0, orig_pred].item())
            )
            fraction_kept += sample_fraction_kept

    n = len(cbm_model.X_test)
    return {
        "pred_overlap": same_pred / n,
        "accuracy": correct / n,
        "confidence_drop_true_class": delta_true / n,
        "confidence_drop_original_pred_class": delta_predclass / n,
        "fraction_kept": fraction_kept / n,
    }


def _average_results(results_list):
    metrics = results_list[0].keys()
    return {
        metric: float(np.mean([result[metric] for result in results_list]))
        for metric in metrics
    }


def _aggregate_attention_maps(explanation, layer_agg="mean"):
    attn_layers = explanation.get("attn_per_layer", None)
    if not attn_layers or all(a is None for a in attn_layers):
        raise ValueError("Explanation does not contain per-layer attention maps.")

    mats = []
    for a in attn_layers:
        if a is None:
            continue
        if isinstance(a, dict):
            a = a.get("temporal", None)
        if a is None:
            continue
        if not torch.is_tensor(a):
            a = torch.as_tensor(a)
        if a.dim() != 3 or a.shape[-1] != a.shape[-2]:
            continue
        mats.append(a.float())

    if len(mats) == 0:
        raise ValueError("No usable [C, T, T] attention maps found in explanation.")

    stack = torch.stack(mats, dim=0)  # [L, C, T, T]
    if layer_agg == "max":
        return stack.max(dim=0).values
    if layer_agg != "mean":
        raise ValueError("layer_agg must be 'mean' or 'max'")
    return stack.mean(dim=0)


def _score_attention_channels(explanation, attention_score_mode="diagonal_mass", layer_agg="mean"):
    attn = _aggregate_attention_maps(explanation, layer_agg=layer_agg)  # [C, T, T]
    if attention_score_mode == "diagonal_mass":
        scores = torch.diagonal(attn, dim1=-2, dim2=-1).mean(dim=-1)
    elif attention_score_mode == "low_entropy":
        probs = attn.clamp(min=1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1).mean(dim=-1)
        scores = -entropy
    elif attention_score_mode == "top_time_mass":
        ti = explanation.get("time_importance", None)
        if ti is None:
            raise ValueError("time_importance is required for attention_score_mode='top_time_mass'")
        if not torch.is_tensor(ti):
            ti = torch.as_tensor(ti)
        ti = ti.float()
        ti = ti / ti.sum().clamp(min=1e-8)
        scores = torch.einsum("t,ctu,u->c", ti, attn, ti)
    else:
        raise ValueError(
            "attention_score_mode must be 'diagonal_mass', 'low_entropy', or 'top_time_mass'"
        )
    return scores.detach().cpu().numpy()


def intervention_topk_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    concept_order_mode="positive",
    intervention_mode="keep_only",
    device="cuda:0",
):
    """
    Generic top-k concept intervention.

    intervention_mode:
        - 'keep_only': keep only top-k important concepts
        - 'remove_topk': remove top-k important concepts

    Assumption:
        model(video, channel_ids=...) ablates the provided channels.
        Therefore:
            - keep_only: ablate all channels except the selected top-k
            - remove_topk: ablate the selected top-k channels
    """
    if intervention_mode not in {"keep_only", "remove_topk"}:
        raise ValueError("intervention_mode must be 'keep_only' or 'remove_topk'")

    model = cbm_model.model
    model.eval()

    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    explanations = base["explanations"]
    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]

    num_concepts = _infer_num_concepts(cbm_model)
    all_concepts = np.arange(num_concepts)

    results = {}

    for k in k_values:
        def selector(i):
            contribs = explanations[i]["concept_contributions_global"]
            ranked = _get_ranked_concepts(contribs, mode=concept_order_mode)
            k_eff = min(k, len(ranked))
            topk = ranked[:k_eff]

            if intervention_mode == "keep_only":
                channel_ids = np.setdiff1d(all_concepts, topk, assume_unique=True)
                fraction_kept = k_eff / num_concepts
            else:  # remove_topk
                channel_ids = topk
                fraction_kept = (num_concepts - k_eff) / num_concepts

            if len(channel_ids) == 0:
                channel_ids = None

            return channel_ids, None, fraction_kept

        results[k] = _compute_metrics_for_intervention(
            cbm_model=cbm_model,
            original_preds=original_preds,
            original_true_probs=original_true_probs,
            original_predclass_probs=original_predclass_probs,
            intervention_selector_fn=selector,
            device=device,
        )

    return results


def keep_only_topk_important_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    concept_order_mode="positive",
    device="cuda:0",
):
    return intervention_topk_concepts(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        k_values=k_values,
        concept_order_mode=concept_order_mode,
        intervention_mode="keep_only",
        device=device,
    )


def remove_topk_important_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    concept_order_mode="positive",
    device="cuda:0",
):
    return intervention_topk_concepts(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        k_values=k_values,
        concept_order_mode=concept_order_mode,
        intervention_mode="remove_topk",
        device=device,
    )


def intervention_random_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    intervention_mode="remove_random",
    num_trials=10,
    random_seed=0,
    device="cuda:0",
):
    """
    Random concept interventions over whole concept channels.

    intervention_mode:
        - 'remove_random': zero k random concept channels across all windows
        - 'insert_random': replace k random concept channels with Gaussian noise
    """
    if intervention_mode not in {"remove_random", "insert_random"}:
        raise ValueError("intervention_mode must be 'remove_random' or 'insert_random'")

    model = cbm_model.model
    model.eval()

    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]

    num_concepts = _infer_num_concepts(cbm_model)
    results = {}

    for k in k_values:
        k_eff = min(k, num_concepts)
        trial_results = []

        for trial_idx in range(num_trials):
            rng = np.random.default_rng(random_seed + 1009 * k + trial_idx)
            torch.manual_seed(random_seed + 1009 * k + trial_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed + 1009 * k + trial_idx)

            def selector(i):
                if k_eff == 0:
                    fraction_kept = 1.0
                    return None, None, fraction_kept, intervention_mode == "insert_random"

                channel_ids = np.sort(rng.choice(num_concepts, size=k_eff, replace=False))
                if intervention_mode == "remove_random":
                    fraction_kept = (num_concepts - k_eff) / num_concepts
                else:
                    fraction_kept = 1.0
                return channel_ids, None, fraction_kept, intervention_mode == "insert_random"

            trial_results.append(
                _compute_metrics_for_intervention(
                    cbm_model=cbm_model,
                    original_preds=original_preds,
                    original_true_probs=original_true_probs,
                    original_predclass_probs=original_predclass_probs,
                    intervention_selector_fn=selector,
                    device=device,
                )
            )

        results[k] = _average_results(trial_results)

    return results


def remove_random_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    num_trials=10,
    random_seed=0,
    device="cuda:0",
):
    return intervention_random_concepts(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        k_values=k_values,
        intervention_mode="remove_random",
        num_trials=num_trials,
        random_seed=random_seed,
        device=device,
    )


def insert_random_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    num_trials=10,
    random_seed=0,
    device="cuda:0",
):
    return intervention_random_concepts(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        k_values=k_values,
        intervention_mode="insert_random",
        num_trials=num_trials,
        random_seed=random_seed,
        device=device,
    )


def remove_topk_important_windows(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    device="cuda:0",
):
    """
    Remove the top-k most important windows, ranked by time_importance.
    """
    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    explanations = base["explanations"]
    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]
    results = {}

    for k in k_values:
        def selector(i):
            time_importance = explanations[i]["time_importance"]
            if not torch.is_tensor(time_importance):
                time_importance = torch.as_tensor(time_importance)
            num_windows = int(time_importance.numel())
            k_eff = min(k, num_windows)
            if k_eff == 0:
                return None, None, 1.0
            ranked_windows = time_importance.argsort(descending=True).detach().cpu().numpy()
            window_ids = ranked_windows[:k_eff]
            return None, window_ids, (num_windows - k_eff) / num_windows

        results[k] = _compute_metrics_for_intervention(
            cbm_model=cbm_model,
            original_preds=original_preds,
            original_true_probs=original_true_probs,
            original_predclass_probs=original_predclass_probs,
            intervention_selector_fn=selector,
            device=device,
        )

    return results


def remove_random_windows(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    num_trials=10,
    random_seed=0,
    device="cuda:0",
):
    """
    Remove k random windows.
    """
    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    explanations = base["explanations"]
    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]
    results = {}

    for k in k_values:
        trial_results = []
        for trial_idx in range(num_trials):
            rng = np.random.default_rng(random_seed + 3001 * k + trial_idx)

            def selector(i):
                time_importance = explanations[i]["time_importance"]
                if not torch.is_tensor(time_importance):
                    time_importance = torch.as_tensor(time_importance)
                num_windows = int(time_importance.numel())
                k_eff = min(k, num_windows)
                if k_eff == 0:
                    return None, None, 1.0
                window_ids = np.sort(rng.choice(num_windows, size=k_eff, replace=False))
                return None, window_ids, (num_windows - k_eff) / num_windows

            trial_results.append(
                _compute_metrics_for_intervention(
                    cbm_model=cbm_model,
                    original_preds=original_preds,
                    original_true_probs=original_true_probs,
                    original_predclass_probs=original_predclass_probs,
                    intervention_selector_fn=selector,
                    device=device,
                )
            )

        results[k] = _average_results(trial_results)

    return results


def leave_one_out_concepts(
    cbm_model,
    explain_instance_fn,
    device="cuda:0",
):
    """
    Remove one concept channel at a time across all windows.
    Returns concept_idx -> metrics.
    """
    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    explanations = base["explanations"]
    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]
    num_concepts = _infer_num_concepts(cbm_model)

    heuristic_scores = np.zeros(num_concepts, dtype=np.float64)
    for explanation in explanations:
        contribs = explanation["concept_contributions_global"]
        if not torch.is_tensor(contribs):
            contribs = torch.as_tensor(contribs)
        heuristic_scores += contribs.detach().cpu().numpy()
    heuristic_scores /= len(explanations)

    results = {}
    for concept_idx in range(num_concepts):
        def selector(i):
            return np.array([concept_idx]), None, (num_concepts - 1) / num_concepts

        metrics = _compute_metrics_for_intervention(
            cbm_model=cbm_model,
            original_preds=original_preds,
            original_true_probs=original_true_probs,
            original_predclass_probs=original_predclass_probs,
            intervention_selector_fn=selector,
            device=device,
        )
        metrics["concept_score"] = float(heuristic_scores[concept_idx])
        results[concept_idx] = metrics

    return results


def leave_one_in_concepts(
    cbm_model,
    explain_instance_fn,
    device="cuda:0",
):
    """
    Keep one concept channel at a time across all windows.
    Returns concept_idx -> metrics.
    """
    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    explanations = base["explanations"]
    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]
    num_concepts = _infer_num_concepts(cbm_model)
    all_concepts = np.arange(num_concepts)

    heuristic_scores = np.zeros(num_concepts, dtype=np.float64)
    for explanation in explanations:
        contribs = explanation["concept_contributions_global"]
        if not torch.is_tensor(contribs):
            contribs = torch.as_tensor(contribs)
        heuristic_scores += contribs.detach().cpu().numpy()
    heuristic_scores /= len(explanations)

    results = {}
    for concept_idx in range(num_concepts):
        def selector(i):
            remove_ids = np.setdiff1d(all_concepts, np.array([concept_idx]), assume_unique=True)
            return remove_ids, None, 1 / num_concepts

        metrics = _compute_metrics_for_intervention(
            cbm_model=cbm_model,
            original_preds=original_preds,
            original_true_probs=original_true_probs,
            original_predclass_probs=original_predclass_probs,
            intervention_selector_fn=selector,
            device=device,
        )
        metrics["concept_score"] = float(heuristic_scores[concept_idx])
        results[concept_idx] = metrics

    return results


def measure_attention_channel_effects(
    cbm_model,
    explain_instance_fn,
    attn_mode="uniform",
    layer_agg="mean",
    attention_score_mode="diagonal_mass",
    device="cuda:0",
):
    """
    Measure the causal effect of neutralizing each attention channel independently.

    Returns a dict: channel_idx -> metrics plus the heuristic attention score.
    """
    model = cbm_model.model
    if not getattr(model, "diagonal_attention", False):
        raise ValueError("Per-channel attention effects require diagonal_attention=True.")

    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    explanations = base["explanations"]
    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]
    num_concepts = _infer_num_concepts(cbm_model)

    heuristic_scores = np.zeros(num_concepts, dtype=np.float64)
    for explanation in explanations:
        heuristic_scores += _score_attention_channels(
            explanation,
            attention_score_mode=attention_score_mode,
            layer_agg=layer_agg,
        )
    heuristic_scores /= len(explanations)

    per_channel_results = {}
    for channel_idx in range(num_concepts):
        def selector(i):
            return {
                "attn_channel_ids": np.array([channel_idx]),
                "attn_mode": attn_mode,
                "fraction_kept": (num_concepts - 1) / num_concepts,
            }

        metrics = _compute_metrics_for_intervention(
            cbm_model=cbm_model,
            original_preds=original_preds,
            original_true_probs=original_true_probs,
            original_predclass_probs=original_predclass_probs,
            intervention_selector_fn=selector,
            device=device,
        )
        metrics["attention_score"] = float(heuristic_scores[channel_idx])
        per_channel_results[channel_idx] = metrics

    return per_channel_results


def intervention_topk_attention_channels(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    attention_score_mode="diagonal_mass",
    attn_mode="uniform",
    layer_agg="mean",
    device="cuda:0",
):
    """
    Intervene on the top-k attention channels, ranked per sample by an attention statistic.
    """
    model = cbm_model.model
    if not getattr(model, "diagonal_attention", False):
        raise ValueError("Per-channel attention interventions require diagonal_attention=True.")

    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    explanations = base["explanations"]
    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]
    num_concepts = _infer_num_concepts(cbm_model)
    results = {}

    for k in k_values:
        def selector(i):
            scores = _score_attention_channels(
                explanations[i],
                attention_score_mode=attention_score_mode,
                layer_agg=layer_agg,
            )
            k_eff = min(k, len(scores))
            if k_eff == 0:
                return {
                    "attn_channel_ids": None,
                    "attn_mode": attn_mode,
                    "fraction_kept": 1.0,
                }
            ranked = np.argsort(scores)[::-1].copy()
            return {
                "attn_channel_ids": ranked[:k_eff].copy(),
                "attn_mode": attn_mode,
                "fraction_kept": (num_concepts - k_eff) / num_concepts,
            }

        results[k] = _compute_metrics_for_intervention(
            cbm_model=cbm_model,
            original_preds=original_preds,
            original_true_probs=original_true_probs,
            original_predclass_probs=original_predclass_probs,
            intervention_selector_fn=selector,
            device=device,
        )

    return results


def intervention_random_attention_channels(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    attn_mode="uniform",
    num_trials=10,
    random_seed=0,
    device="cuda:0",
):
    """
    Intervene on k random attention channels.
    """
    model = cbm_model.model
    if not getattr(model, "diagonal_attention", False):
        raise ValueError("Per-channel attention interventions require diagonal_attention=True.")

    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]
    num_concepts = _infer_num_concepts(cbm_model)
    results = {}

    for k in k_values:
        k_eff = min(k, num_concepts)
        trial_results = []

        for trial_idx in range(num_trials):
            rng = np.random.default_rng(random_seed + 2003 * k + trial_idx)

            def selector(i):
                if k_eff == 0:
                    return {
                        "attn_channel_ids": None,
                        "attn_mode": attn_mode,
                        "fraction_kept": 1.0,
                    }
                attn_channel_ids = np.sort(rng.choice(num_concepts, size=k_eff, replace=False))
                return {
                    "attn_channel_ids": attn_channel_ids,
                    "attn_mode": attn_mode,
                    "fraction_kept": (num_concepts - k_eff) / num_concepts,
                }

            trial_results.append(
                _compute_metrics_for_intervention(
                    cbm_model=cbm_model,
                    original_preds=original_preds,
                    original_true_probs=original_true_probs,
                    original_predclass_probs=original_predclass_probs,
                    intervention_selector_fn=selector,
                    device=device,
                )
            )

        results[k] = _average_results(trial_results)

    return results


def intervention_topk_local_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    concept_order_mode="positive",
    intervention_mode="keep_only",
    device="cuda:0",
):
    """
    Top-k local concept intervention over individual (window, concept) pairs.

    intervention_mode:
        - 'keep_only': keep only the top-k local concept slots
        - 'remove_topk': remove the top-k local concept slots

    Local importance is taken from explanation["concept_contributions_per_time"].
    """
    if intervention_mode not in {"keep_only", "remove_topk"}:
        raise ValueError("intervention_mode must be 'keep_only' or 'remove_topk'")

    model = cbm_model.model
    model.eval()

    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    explanations = base["explanations"]
    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]

    results = {}

    for k in k_values:
        def selector(i):
            contribs_per_time = explanations[i]["concept_contributions_per_time"]
            num_windows, num_concepts = contribs_per_time.shape
            total_pairs = num_windows * num_concepts
            k_eff = min(k, total_pairs)

            ranked_windows, ranked_channels = _get_ranked_local_concepts(
                contribs_per_time,
                mode=concept_order_mode,
            )

            top_windows = ranked_windows[:k_eff]
            top_channels = ranked_channels[:k_eff]

            if intervention_mode == "keep_only":
                keep_mask = np.zeros(total_pairs, dtype=bool)
                keep_mask[:k_eff] = True
                remove_windows = ranked_windows[~keep_mask]
                remove_channels = ranked_channels[~keep_mask]
                fraction_kept = k_eff / total_pairs
            else:  # remove_topk
                remove_windows = top_windows
                remove_channels = top_channels
                fraction_kept = (total_pairs - k_eff) / total_pairs

            if len(remove_windows) == 0:
                return None, None, fraction_kept

            return remove_channels, remove_windows, fraction_kept

        results[k] = _compute_metrics_for_intervention(
            cbm_model=cbm_model,
            original_preds=original_preds,
            original_true_probs=original_true_probs,
            original_predclass_probs=original_predclass_probs,
            intervention_selector_fn=selector,
            device=device,
        )

    return results


def keep_only_topk_important_local_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    concept_order_mode="positive",
    device="cuda:0",
):
    return intervention_topk_local_concepts(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        k_values=k_values,
        concept_order_mode=concept_order_mode,
        intervention_mode="keep_only",
        device=device,
    )


def remove_topk_important_local_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    concept_order_mode="positive",
    device="cuda:0",
):
    return intervention_topk_local_concepts(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        k_values=k_values,
        concept_order_mode=concept_order_mode,
        intervention_mode="remove_topk",
        device=device,
    )


def intervention_random_local_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    intervention_mode="remove_random",
    num_trials=10,
    random_seed=0,
    device="cuda:0",
):
    """
    Random local concept interventions over individual (window, concept) pairs.

    intervention_mode:
        - 'remove_random': zero k random local concept slots
        - 'insert_random': replace k random local concept slots with Gaussian noise
    """
    if intervention_mode not in {"remove_random", "insert_random"}:
        raise ValueError("intervention_mode must be 'remove_random' or 'insert_random'")

    model = cbm_model.model
    model.eval()

    base = compute_original_outputs(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        device=device,
    )

    explanations = base["explanations"]
    original_preds = base["original_preds"]
    original_true_probs = base["original_true_probs"]
    original_predclass_probs = base["original_predclass_probs"]

    results = {}

    for k in k_values:
        trial_results = []

        for trial_idx in range(num_trials):
            rng = np.random.default_rng(random_seed + 1009 * k + trial_idx)
            torch.manual_seed(random_seed + 1009 * k + trial_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed + 1009 * k + trial_idx)

            def selector(i):
                contribs_per_time = explanations[i]["concept_contributions_per_time"]
                num_windows, num_concepts = contribs_per_time.shape
                total_pairs = num_windows * num_concepts
                k_eff = min(k, total_pairs)

                if k_eff == 0:
                    fraction_kept = 1.0
                    return None, None, fraction_kept, intervention_mode == "insert_random"

                flat_ids = np.sort(rng.choice(total_pairs, size=k_eff, replace=False))
                window_ids, channel_ids = np.unravel_index(flat_ids, (num_windows, num_concepts))
                if intervention_mode == "remove_random":
                    fraction_kept = (total_pairs - k_eff) / total_pairs
                else:
                    fraction_kept = 1.0

                return channel_ids, window_ids, fraction_kept, intervention_mode == "insert_random"

            trial_results.append(
                _compute_metrics_for_intervention(
                    cbm_model=cbm_model,
                    original_preds=original_preds,
                    original_true_probs=original_true_probs,
                    original_predclass_probs=original_predclass_probs,
                    intervention_selector_fn=selector,
                    device=device,
                )
            )

        results[k] = _average_results(trial_results)

    return results


def remove_random_local_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    num_trials=10,
    random_seed=0,
    device="cuda:0",
):
    return intervention_random_local_concepts(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        k_values=k_values,
        intervention_mode="remove_random",
        num_trials=num_trials,
        random_seed=random_seed,
        device=device,
    )


def insert_random_local_concepts(
    cbm_model,
    explain_instance_fn,
    k_values=(1, 2, 3, 5, 10),
    num_trials=10,
    random_seed=0,
    device="cuda:0",
):
    return intervention_random_local_concepts(
        cbm_model=cbm_model,
        explain_instance_fn=explain_instance_fn,
        k_values=k_values,
        intervention_mode="insert_random",
        num_trials=num_trials,
        random_seed=random_seed,
        device=device,
    )


def plot_intervention_results(
    results,
    metrics=None,
    title="Intervention Results",
    xlabel="k",
    figsize=(8, 5),
    marker="o",
):
    if metrics is None:
        metrics = list(next(iter(results.values())).keys())

    ks = sorted(results.keys())

    plt.figure(figsize=figsize)
    for metric in metrics:
        ys = [results[k][metric] for k in ks]
        plt.plot(ks, ys, marker=marker, label=metric)

    plt.xlabel(xlabel)
    plt.ylabel("value")
    plt.title(title)
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_compare_interventions(
    results_keep,
    results_remove,
    metric="confidence_drop_original_pred_class",
    label_a="keep_only_topk",
    label_b="remove_topk",
    title=None,
    figsize=(8, 5),
    marker="o",
):
    ks_keep = sorted(results_keep.keys())
    ks_remove = sorted(results_remove.keys())

    plt.figure(figsize=figsize)
    plt.plot(
        ks_keep,
        [results_keep[k][metric] for k in ks_keep],
        marker=marker,
        label=label_a,
    )
    plt.plot(
        ks_remove,
        [results_remove[k][metric] for k in ks_remove],
        marker=marker,
        label=label_b,
    )

    plt.xlabel("k")
    plt.ylabel(metric)
    if title is None:
        title = f"Comparison of interventions: {metric}"
    plt.title(title)
    plt.xticks(sorted(set(ks_keep).union(ks_remove)))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
