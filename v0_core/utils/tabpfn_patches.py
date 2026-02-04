import torch
import logging
import numpy as np
try:
    from tabpfn import TabPFNClassifier
    from tabpfn.base import create_inference_engine, determine_precision
    from tabpfn.utils import infer_random_state
    from tabpfn.classifier import _validate_eval_metric
    from tabpfn.inference import InferenceEngineBatchedNoPreprocessing
except ImportError as e:
    print(f"导入 TabPFN 模块失败: {e}")
    print("请确保已安装 tabpfn，并且处于包含 tabpfn 源代码的环境中。")
    exit(1)

def fixed_fit(self, X, y) -> "TabPFNClassifier":
    """修复 fit 方法：解决 differentiable_input=True 时 ensemble_configs 未定义的问题"""
    self.eval_metric_ = _validate_eval_metric(self.eval_metric)

    if self.fit_mode == "batched":
        logging.warning("Switching from 'batched' to 'fit_preprocessors' mode...")
        self.fit_mode = "fit_preprocessors"

    if not hasattr(self, "models_") or not self.differentiable_input:
        byte_size, rng = self._initialize_model_variables()
        ensemble_configs, X, y = self._initialize_dataset_preprocessing(X, y, rng)
    else:
        _, rng = infer_random_state(self.random_state)
        _, _, byte_size = determine_precision(self.inference_precision, self.devices_)
        ensemble_configs, X, y = self._initialize_dataset_preprocessing(X, y, rng)

    self._maybe_calibrate_temperature_and_tune_decision_thresholds(X=X, y=y)

    self.executor_ = create_inference_engine(
        X_train=X,
        y_train=y,
        models=self.models_,
        ensemble_configs=ensemble_configs,
        cat_ix=self.inferred_categorical_indices_,
        fit_mode=self.fit_mode,
        devices_=self.devices_,
        rng=rng,
        n_preprocessing_jobs=self.n_preprocessing_jobs,
        byte_size=byte_size,
        forced_inference_dtype_=self.forced_inference_dtype_,
        memory_saving_mode=self.memory_saving_mode,
        use_autocast_=self.use_autocast_,
        inference_mode=not self.differentiable_input,
    )
    return self

def fixed_forward(
    self,
    X: list[torch.Tensor] | torch.Tensor,
    *,
    use_inference_mode: bool = False,
    return_logits: bool = False,
    return_raw_logits: bool = False,
) -> torch.Tensor:
    """修复 forward 方法：允许 standard inference 下保留梯度"""
    if return_logits and return_raw_logits:
        raise ValueError("Cannot return both logits and raw logits.")

    is_standard_inference = not isinstance(
        self.executor_, InferenceEngineBatchedNoPreprocessing
    )
    is_batched_for_grads = (
        not use_inference_mode
        and isinstance(self.executor_, InferenceEngineBatchedNoPreprocessing)
        and isinstance(X, list)
    )

    assert is_standard_inference or is_batched_for_grads, "Invalid forward pass."

    if self.fit_mode in ["fit_preprocessors", "batched"]:
        self.executor_.use_torch_inference_mode(use_inference=use_inference_mode)

    outputs = []
    for output, config in self.executor_.iter_outputs(X, autocast=self.use_autocast_):
        processed_output = output.unsqueeze(1) if output.ndim == 2 else output
        config_list = [config] if output.ndim == 2 else config
        
        output_batch = []
        for i, batch_config in enumerate(config_list):
            if batch_config.class_permutation is None:
                output_batch.append(processed_output[:, i, : self.n_classes_])
            else:
                use_perm = batch_config.class_permutation
                if len(use_perm) != self.n_classes_:
                    full_perm = np.arange(self.n_classes_)
                    full_perm[:len(use_perm)] = use_perm
                    use_perm = full_perm
                output_batch.append(processed_output[:, i, use_perm])
        outputs.append(torch.stack(output_batch, dim=1))

    stacked_outputs = torch.stack(outputs) # (Chunks, Samples, Est, Classes)

    if return_logits:
        temp_scaled = self._apply_temperature(stacked_outputs)
        output = temp_scaled.mean(dim=(0, 2)) 
    elif return_raw_logits:
        output = stacked_outputs
    else:
        temp_scaled = self._apply_temperature(stacked_outputs)
        avg_logits = temp_scaled.mean(dim=(0, 2))
        output = torch.nn.functional.softmax(avg_logits, dim=-1)

    if not use_inference_mode:
        if return_logits and output.ndim == 2:
            return output
        if output.ndim == 2:
            output = output.unsqueeze(0)
        output = output.transpose(0, 1).transpose(1, 2)
    elif output.ndim > 2 and use_inference_mode:
        output = output.squeeze(1) if not return_raw_logits else output.squeeze(2)

    return output