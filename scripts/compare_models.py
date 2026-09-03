import argparse
import csv
import os
import time
import numpy as np
import onnxruntime as ort

from utils import load_input_array, collect_input_paths


def _require_single_input(session: ort.InferenceSession, model_path: str):
    """Return the sole model input, or raise if the network is not single-input."""
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(
            f"Expected exactly one input in {model_path}, found {len(inputs)}"
        )
    return inputs[0]


def _input_shape_from_onnx(model_path: str) -> tuple[int, ...]:
    """Read the concrete input shape from an ONNX model (single input only)."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = _require_single_input(session, model_path)
    # Dynamic dims (e.g. batch) cannot form a concrete numpy shape; use 1.
    shape = [
        1 if isinstance(dim, str) or dim is None else int(dim) for dim in inp.shape
    ]
    return tuple(shape)


def _as_class_outputs(float_out: np.ndarray, quant_out: np.ndarray):
    """Reshape model outputs to (batch, num_classes), last axis = classes."""
    float_probs = float_out.reshape(-1, float_out.shape[-1])
    quant_probs = quant_out.reshape(-1, quant_out.shape[-1])
    assert (
        float_probs.shape == quant_probs.shape
    ), f"Output shape mismatch: float {float_out.shape} vs quant {quant_out.shape}"
    return float_probs, quant_probs


def _kl_divergence(float_out: np.ndarray, quant_out: np.ndarray) -> float:
    """Mean KL(float || quant) over the batch (nats).

    Assumes the last layer is already a softmax (outputs are class probabilities).
    """
    p, q = _as_class_outputs(float_out, quant_out)
    # Avoid log(0); tiny mass does not change the comparison meaningfully.
    eps = 1e-12
    kl = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)
    return float(np.mean(kl))


def _topk_class_matches(float_out: np.ndarray, quant_out: np.ndarray, ks=(1, 5, 10)):
    """Count overlapping top-k class indices between float and quantized outputs.

    Treats the last axis as the class dimension. Returns the mean overlap count
    over the batch for each k (e.g. top-5 match is in [0, 5]).
    """
    float_probs, quant_probs = _as_class_outputs(float_out, quant_out)

    matches = {}
    for k in ks:
        assert (
            float_probs.shape[-1] >= k
        ), f"Requested top-{k} but model only has {float_probs.shape[-1]} classes"
        per_sample = []
        for f_row, q_row in zip(float_probs, quant_probs):
            top_f = set(np.argpartition(f_row, -k)[-k:])
            top_q = set(np.argpartition(q_row, -k)[-k:])
            per_sample.append(len(top_f & top_q))
        matches[k] = float(np.mean(per_sample))
    return matches


def _accuracy_metrics(float_out: np.ndarray, quant_out: np.ndarray):
    """Return MAE, max abs diff, cosine sim, top-k matches, and KL for one pair."""
    mae = float(np.mean(np.abs(float_out - quant_out)))
    max_diff = float(np.max(np.abs(float_out - quant_out)))

    flat_float = float_out.flatten()
    flat_quant = quant_out.flatten()
    cosine_sim = float(
        np.dot(flat_float, flat_quant)
        / (np.linalg.norm(flat_float) * np.linalg.norm(flat_quant))
    )

    topk_matches = _topk_class_matches(float_out, quant_out)
    kl_div = _kl_divergence(float_out, quant_out)
    return mae, max_diff, cosine_sim, topk_matches, kl_div


def _compare_models(
    float_model_path: str,
    quant_model_path: str,
    input_arrays: list[np.ndarray],
):
    print("Loading models into ONNX Runtime...")
    providers = ["CPUExecutionProvider"]

    sess_float = ort.InferenceSession(float_model_path, providers=providers)
    sess_quant = ort.InferenceSession(quant_model_path, providers=providers)

    input_meta = _require_single_input(sess_float, float_model_path)
    _require_single_input(sess_quant, quant_model_path)
    input_name = input_meta.name

    # Latency: benchmark once on the first validation input.
    latency_dict = {input_name: input_arrays[0]}
    _ = sess_float.run(None, latency_dict)
    start_time = time.perf_counter()
    for _ in range(100):
        _ = sess_float.run(None, latency_dict)
    float_latency = (time.perf_counter() - start_time) / 100 * 1000  # convert to ms

    _ = sess_quant.run(None, latency_dict)
    start_time = time.perf_counter()
    for _ in range(100):
        _ = sess_quant.run(None, latency_dict)
    quant_latency = (time.perf_counter() - start_time) / 100 * 1000  # convert to ms

    # Accuracy: accumulate means over the full validation set.
    mae_sum = 0.0
    max_diff_sum = 0.0
    cosine_sum = 0.0
    kl_sum = 0.0
    topk_sums = {1: 0.0, 5: 0.0, 10: 0.0}
    n = len(input_arrays)

    for input_arr in input_arrays:
        input_dict = {input_name: input_arr}
        float_out = sess_float.run(None, input_dict)[0]
        quant_out = sess_quant.run(None, input_dict)[0]
        mae, max_diff, cosine_sim, topk_matches, kl_div = _accuracy_metrics(
            float_out, quant_out
        )
        mae_sum += mae
        max_diff_sum += max_diff
        cosine_sum += cosine_sim
        kl_sum += kl_div
        for k in topk_sums:
            topk_sums[k] += topk_matches[k]

    mae = mae_sum / n
    max_diff = max_diff_sum / n
    cosine_sim = cosine_sum / n
    kl_div = kl_sum / n
    topk_matches = {k: topk_sums[k] / n for k in topk_sums}

    # ==========================================
    # Print Results
    # ==========================================
    speedup = float_latency / quant_latency
    print("\n" + "=" * 40)
    print(" LATENCY")
    print("=" * 40)
    print(f"Float32 Latency:   {float_latency:.2f} ms")
    print(f"Int8 Latency:      {quant_latency:.2f} ms")
    print(f"Speedup factor:    {speedup:.2f}x")
    print("\n" + "=" * 40)
    print(f" ACCURACY (mean over {n} input(s))")
    print("=" * 40)
    print(f"Mean Absolute Error (MAE):    {mae:.6f}")
    print(f"Max Absolute Difference:      {max_diff:.6f}")
    print(f"Cosine Similarity (0 to 1):   {cosine_sim:.6f}")
    print(f"KL(float || quant) [nats]:    {kl_div:.6f}")
    print(f"Top-1 class matches:          {topk_matches[1]:.2f} / 1")
    print(f"Top-5 class matches:          {topk_matches[5]:.2f} / 5")
    print(f"Top-10 class matches:         {topk_matches[10]:.2f} / 10")

    return speedup, mae, max_diff, cosine_sim, topk_matches, kl_div


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare quantized model against original float model."
    )
    parser.add_argument("float_model", type=str, help="original float model")
    parser.add_argument(
        "quant_model",
        type=str,
        help="quantized model",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to an input file or a directory of validation images",
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="quantization method type",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_file",
        type=str,
        help="output file with results",
        default="perf_info.csv",
    )

    return parser.parse_args()


def _main():
    args = _parse_args()

    shape = _input_shape_from_onnx(args.float_model)
    input_paths = collect_input_paths(args.input)
    print(f"Scoring {len(input_paths)} validation input(s) from {args.input}")
    input_arrays = [
        load_input_array(path, shape, dtype=np.float32) for path in input_paths
    ]
    speedup, mae, max_diff, cosine_sim, topk_matches, kl_div = _compare_models(
        args.float_model, args.quant_model, input_arrays
    )

    results = {
        "speedup": speedup,
        "mean_absolute_error": mae,
        "max_absolute_error": max_diff,
        "cosine_sim": cosine_sim,
        "kl_divergence": kl_div,
        "top1_matches": topk_matches[1],
        "top5_matches": topk_matches[5],
        "top10_matches": topk_matches[10],
        "type": args.type,
    }

    is_empty = (
        not os.path.isfile(args.output_file) or os.path.getsize(args.output_file) == 0
    )
    with open(args.output_file, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if is_empty:
            writer.writeheader()
        writer.writerow(results)


if __name__ == "__main__":
    _main()
