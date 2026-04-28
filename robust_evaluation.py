import os
import time
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from core.dataset import MMDataEvaluationLoader
from models.TFMamba import build_model
from core.metric import MetricsTop

try:
    from jtop import jtop  # type: ignore
    JTOP_AVAILABLE = True
except Exception:
    jtop = None
    JTOP_AVAILABLE = False

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class PerformanceTracker:
    def __init__(self):
        self.latency_list: List[float] = []
        self.gpu_pwr_list: List[float] = []
        self.temp_list: List[float] = []
        self.mem_allocated_max: int = 0

    def update_memory(self):
        if USE_CUDA:
            self.mem_allocated_max = max(
                self.mem_allocated_max,
                torch.cuda.max_memory_allocated()
            )

    def update_jetson(self, jetson_obj: Any):
        if jetson_obj is None:
            return
        try:
            if not jetson_obj.ok():
                return
            power = jetson_obj.power
            if isinstance(power, dict) and "tot" in power and "avg" in power["tot"]:
                self.gpu_pwr_list.append(float(power["tot"]["avg"]) / 1000.0)
            temp_data = jetson_obj.temperature
            if isinstance(temp_data, dict):
                if "tj" in temp_data and isinstance(temp_data["tj"], dict):
                    self.temp_list.append(float(temp_data["tj"].get("temp", 0.0)))
                elif "cpu" in temp_data and isinstance(temp_data["cpu"], dict):
                    self.temp_list.append(float(temp_data["cpu"].get("temp", 0.0)))
        except Exception:
            pass

    def report(
        self,
        batch_size: int,
        final_metrics: Dict[str, float],
        missing_rate: float,
        eval_missing_modalities: str,
        missing_policy: str,
        ckpt_path: str,
    ):
        avg_latency_ms = float(np.mean(self.latency_list) * 1000.0) if self.latency_list else 0.0
        throughput = float(batch_size / (avg_latency_ms / 1000.0)) if avg_latency_ms > 0 else 0.0
        avg_power = float(np.mean(self.gpu_pwr_list)) if self.gpu_pwr_list else -1.0
        avg_temp = float(np.mean(self.temp_list)) if self.temp_list else -1.0

        print("\n" + "=" * 78)
        print("🛡️  CMS-Mamba 极限生存评估报告")
        print("-" * 78)
        print(f"📦 Checkpoint             : {ckpt_path}")
        print(f"🎯 Missing Rate           : {missing_rate:.1f}")
        print(f"🎯 Missing Modalities     : {eval_missing_modalities}")
        print(f"🧊 Missing Policy         : {missing_policy}")
        print(f"📊 Batch Size             : {batch_size}")
        print("-" * 78)
        print("【边缘/推理性能】")
        print(f"⏱️  平均 Batch 推理延迟     : {avg_latency_ms:.2f} ms")
        print(f"🚀 估算吞吐量              : {throughput:.2f} samples/s")
        print(f"💾 CUDA 显存峰值           : {self.mem_allocated_max / 1024 ** 2:.2f} MB")
        print(f"⚡ 平均运行功耗            : {avg_power:.2f} W" if avg_power >= 0 else "⚡ 平均运行功耗            : N/A")
        print(f"🔥 平均核心温度            : {avg_temp:.2f} °C" if avg_temp >= 0 else "🔥 平均核心温度            : N/A")
        print("-" * 78)
        print("【鲁棒性指标】")
        for k, v in final_metrics.items():
            print(f"✨ {k:<16}: {v:.4f}")
        print("=" * 78 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="CMS-Mamba robust evaluation for one missing rate.")
    parser.add_argument("--config_file", type=str, default="configs/eval_mosei.yaml")
    parser.add_argument("--ckpt_path", type=str, default="ckpt/mosei/best_MAE_1111.pth")
    parser.add_argument("--missing_rate", type=float, default=1.0)
    parser.add_argument(
        "--eval_missing_modalities",
        type=str,
        default="TAV",
        choices=["T", "A", "V", "TA", "TV", "AV", "TAV"],
        help="Which modalities are corrupted during evaluation.",
    )
    parser.add_argument(
        "--missing_policy",
        type=str,
        default="zero",
        choices=["zero", "model"],
        help="zero = clear learned missing tokens; model = keep learned missing tokens.",
    )
    parser.add_argument("--strict_mask_check", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--disable_jtop", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    return parser.parse_args()


def load_config(config_file: str, missing_rate: float, eval_missing_modalities: str, num_workers: Optional[int]):
    with open(config_file, "r", encoding="utf-8") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        
    args.setdefault("base", {})
    args.setdefault("dataset", {})
    
    # 全局覆盖所有可能的 missing_rate 字段，防止 Dataloader 读错
    args["base"]["missing_rate_eval_test"] = float(missing_rate)
    args["base"]["missing_rate"] = float(missing_rate)
    args["dataset"]["missing_rate"] = float(missing_rate)
    
    args["base"]["eval_missing_modalities"] = eval_missing_modalities
    
    if num_workers is not None:
        args["base"]["num_workers"] = int(num_workers)
        
    return args


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        elif k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 找不到权重文件: {ckpt_path}")
    print(f"📦 正在加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_info = model.load_state_dict(normalize_state_dict(state_dict), strict=False)
    print(f"⚠️ [诊断信息] Missing Keys 数量: {len(load_info.missing_keys)}")
    if load_info.missing_keys:
        print(f"🚨 Missing Keys 前 10 个: {load_info.missing_keys[:10]}")
    print(f"⚠️ [诊断信息] Unexpected Keys 数量: {len(load_info.unexpected_keys)}")
    if load_info.unexpected_keys:
        print(f"🚨 Unexpected Keys 前 10 个: {load_info.unexpected_keys[:10]}")
    if not load_info.missing_keys and not load_info.unexpected_keys:
        print("✅ 权重加载完成：没有缺失或多余参数。")


def apply_missing_policy(model: torch.nn.Module, missing_policy: str):
    if missing_policy == "model":
        print("[missing-policy] model: 保留 checkpoint 中的 learned missing token。")
        return
    if missing_policy == "zero":
        changed = []
        with torch.no_grad():
            if hasattr(model, "v_mask_token"):
                model.v_mask_token.zero_()
                changed.append("v_mask_token")
            if hasattr(model, "a_mask_token"):
                model.a_mask_token.zero_()
                changed.append("a_mask_token")
        print(f"[missing-policy] zero: 已将 {changed} 置零。")
        return
    raise ValueError(f"Unknown missing_policy: {missing_policy}")


def check_first_batch_mask(batch_data: Dict[str, Any], missing_rate: float, eval_missing_modalities: str, strict: bool):
    messages = []
    mask_keys = {"T": "text_missing_mask", "A": "audio_missing_mask", "V": "vision_missing_mask"}
    orig_keys = {"T": "text_mask", "A": "audio_mask", "V": "vision_mask"}
    
    for modality, key in mask_keys.items():
        if key not in batch_data:
            continue
        
        missing_mask = batch_data[key]
        orig_mask = batch_data[orig_keys[modality]]
        
        # 计算真实的有效 token 数量（剔除 Padding）
        valid_count = float(orig_mask.sum().item())
        if valid_count > 0:
            keep_count = float(missing_mask.sum().item())
            keep_ratio = keep_count / valid_count
        else:
            keep_ratio = 1.0
            
        missing_ratio = 1.0 - keep_ratio
        messages.append(f"{modality}: missing={missing_ratio:.4f}, keep={keep_ratio:.4f}")
        
        target = float(missing_rate) if modality in eval_missing_modalities else 0.0
        # 文本模态默认保留 CLS 和 SEP，所以 1.0 缺失率时允许一定误差
        tolerance = 0.12 if modality == "T" else 0.08
        
        if strict and abs(missing_ratio - target) > tolerance:
            raise RuntimeError(
                f"[strict-mask-check] {modality} missing ratio mismatch: "
                f"target={target:.4f}, actual={missing_ratio:.4f}, "
                f"eval_missing_modalities={eval_missing_modalities}"
            )
            
    print(
        f"[mask-check] target_r={missing_rate:.1f}, "
        f"eval_missing_modalities={eval_missing_modalities}, "
        + ", ".join(messages)
    )


def model_forward(model: torch.nn.Module, batch_data: Dict[str, Any], disable_amp: bool):
    v_m = batch_data["vision_m"].to(DEVICE)
    a_m = batch_data["audio_m"].to(DEVICE)
    t_m = batch_data["text_m"].to(DEVICE)
    if USE_CUDA and not disable_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model((None, None, None), (v_m, a_m, t_m))
    return model((None, None, None), (v_m, a_m, t_m))


def evaluate_one_missing_rate():
    opt = parse_args()
    print(f"Using device: {DEVICE}")
    print(opt)
    if opt.missing_rate < 0.0 or opt.missing_rate > 1.0:
        raise ValueError("--missing_rate must be between 0.0 and 1.0")

    args = load_config(opt.config_file, opt.missing_rate, opt.eval_missing_modalities, opt.num_workers)
    print(args)
    print(f"⚠️ 当前缺失率测试: {opt.missing_rate:.1f}")
    print(f"🎯 缺失模态: {opt.eval_missing_modalities}")
    print(f"🧊 缺失策略: {opt.missing_policy}")

    dataset_name = args["dataset"]["datasetName"]
    batch_size = int(args["base"]["batch_size"])

    model = build_model(args).to(DEVICE)
    load_checkpoint(model, opt.ckpt_path)
    apply_missing_policy(model, opt.missing_policy)
    model.eval()

    metrics_handler = MetricsTop(train_mode=args["base"]["train_mode"]).getMetics(dataset_name)
    data_loader = MMDataEvaluationLoader(args)

    tracker = PerformanceTracker()
    all_preds = []
    all_labels = []

    use_jtop = JTOP_AVAILABLE and not opt.disable_jtop
    print("✅ jtop 可用：将记录 Jetson 功耗与温度。" if use_jtop else "ℹ️ jtop 不可用或已禁用：仅记录延迟、吞吐量与显存。")
    jetson_context = jtop() if use_jtop else None

    def run_loop(jetson_obj=None):
        with torch.no_grad():
            for i, batch_data in enumerate(data_loader):
                if i == 0:
                    check_first_batch_mask(batch_data, opt.missing_rate, opt.eval_missing_modalities, opt.strict_mask_check)
                labels = batch_data["labels"]["M"].to(DEVICE)
                if USE_CUDA:
                    torch.cuda.reset_peak_memory_stats()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    out = model_forward(model, batch_data, opt.disable_amp)
                    end_event.record()
                    torch.cuda.synchronize()
                    latency_s = start_event.elapsed_time(end_event) / 1000.0
                else:
                    start = time.perf_counter()
                    out = model_forward(model, batch_data, opt.disable_amp)
                    latency_s = time.perf_counter() - start

                tracker.latency_list.append(latency_s)
                tracker.update_memory()
                tracker.update_jetson(jetson_obj)
                all_preds.append(out["sentiment_preds"].float().cpu())
                all_labels.append(labels.float().cpu())

                if (i + 1) % 10 == 0:
                    print(f"进度: 已处理 {i + 1} 个 Batch...")

    if jetson_context is not None:
        with jetson_context as jetson:
            run_loop(jetson)
    else:
        run_loop(None)

    final_preds = torch.cat(all_preds, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    results = metrics_handler(final_preds, final_labels)
    tracker.report(batch_size, results, opt.missing_rate, opt.eval_missing_modalities, opt.missing_policy, opt.ckpt_path)


if __name__ == "__main__":
    evaluate_one_missing_rate()