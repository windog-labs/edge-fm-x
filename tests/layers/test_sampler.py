"""
Sampler 层的正确性和性能测试（pytest 单元测试）

测试 edge_fm.SamplerLayer 的实现，使用 flashinfer 作为参考实现。
包括：
1. 正确性测试：测试不同 temperature 值
2. 性能测试：确保性能不低于 flashinfer 的 0.85 倍
3. Greedy 测试：temperature=0 时使用 topKStage1Greedy + topKStage2Greedy，验证与 torch.argmax / Transformers 一致
"""

import json
import numpy as np
import torch
import pytest
import tempfile
import os
import statistics
from pathlib import Path

from tests.layers._test_utils import make_layer_engine_config

import edge_fm
import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def create_sampler_layer(vocab_size, temperature=1.0, seed=42):
    """创建 SamplerLayer 实例"""
    # 创建临时目录用于存放 config.json
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "config.json")
    
    # 创建模型配置文件
    model_config = {
        "vocab_size": vocab_size
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f)
    
    # 创建 engine_config.json 文件
    engine_config_dir = tempfile.mkdtemp()
    engine_config_path = os.path.join(engine_config_dir, "engine_config.json")
    engine_config = make_layer_engine_config(temp_dir, with_operator_table=False)
    engine_config["sampling"] = {
        "temperature": temperature,
        "seed": seed,
    }
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f)
    
    return edge_fm.SamplerLayer(engine_config_path)


def tensor_to_edge_fm_tensor(torch_tensor):
    """将 PyTorch tensor 转换为 edge_fm.Tensor（通过 DLPack）
    需确保 tensor 是 contiguous 且 row-major。对已 contiguous 的 tensor，
    .contiguous() 是空操作；对 view/transpose 等则会产生连续拷贝。
    """
    capsule = torch_tensor.contiguous().__dlpack__()
    return edge_fm.Tensor.from_dlpack(capsule)


class TestSampler:
    """Sampler 层测试类（包括正确性和性能测试）"""
    
    @pytest.mark.order(1)  # 使用order保证性能测试时(order2)memorypool已经初始化
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [32000, 50000])
    @pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5])
    def test_forward_correctness(self, batch_size, vocab_size, temperature):
        """测试采样前向传播的正确性
        
        参数:
            batch_size: 批次大小
            vocab_size: 词汇表大小
            temperature: 温度参数
        """
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 创建输入 logits
        logits = torch.randn(batch_size, vocab_size, device="cuda:0", dtype=torch.float32)
        
        # 使用 FlashInfer 作为参考实现: 应用 temperature scaling 然后 softmax，最后采样
        seed = 42
        logits_scaled = logits / temperature
        probs = torch.softmax(logits_scaled, dim=-1)
        token_ids_flashinfer = flashinfer.sampling.sampling_from_probs(
            probs, seed=seed, offset=0
        )
        
        # 使用 edge_fm 实现（使用相同的 seed）
        layer = create_sampler_layer(vocab_size, temperature, seed=seed)
        
        logits_efm = tensor_to_edge_fm_tensor(logits)
        token_ids_efm = tensor_to_edge_fm_tensor(
            torch.zeros(batch_size, device="cuda:0", dtype=torch.int32)
        )
        
        # 执行 forward
        layer.forward(logits_efm, token_ids_efm)
        token_ids_efm_torch = torch.from_dlpack(token_ids_efm.to_dlpack())
        torch.cuda.synchronize()
        
        # 验证结果完全一致（因为使用了相同的 seed）
        assert torch.equal(token_ids_efm_torch, token_ids_flashinfer), (
            f"结果不完全一致 (batch_size={batch_size}, vocab_size={vocab_size}, "
            f"temperature={temperature}, seed={seed})\n"
            f"EdgeFM: {token_ids_efm_torch}\n"
            f"FlashInfer: {token_ids_flashinfer}"
        )
    
    @pytest.mark.order(1)
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [32000, 50257, 151936])
    def test_greedy_argmax_correctness(self, batch_size, vocab_size):
        """测试 temperature=0 (greedy argmax) 的正确性

        当 temperature < 1e-6 时，SamplerLayer 使用 topKStage1Greedy + topKStage2Greedy
        而非 FlashInfer SamplingFromProb。此测试验证 greedy 路径与
        torch.argmax 结果完全一致。
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        logits = torch.randn(batch_size, vocab_size, device="cuda:0", dtype=torch.float32)

        expected = torch.argmax(logits, dim=-1).to(torch.int32)

        layer = create_sampler_layer(vocab_size, temperature=0.0, seed=0)
        logits_efm = tensor_to_edge_fm_tensor(logits)
        token_ids_efm = tensor_to_edge_fm_tensor(
            torch.zeros(batch_size, device="cuda:0", dtype=torch.int32)
        )
        layer.forward(logits_efm, token_ids_efm)
        torch.cuda.synchronize()
        result = torch.from_dlpack(token_ids_efm.to_dlpack())

        assert torch.equal(result, expected), (
            f"Greedy argmax 结果与 torch.argmax 不一致\n"
            f"  batch_size={batch_size}, vocab_size={vocab_size}\n"
            f"  EdgeFM:  {result}\n"
            f"  Expected: {expected}"
        )

    @pytest.mark.order(1)
    def test_greedy_argmax_edge_cases(self):
        """测试 greedy 路径 (topKStage1/2) 的边界情况

        覆盖：最大值在首位、末位、vocab_size 不是 256 的倍数、
        存在大量相同值、极端值等场景。
        """
        vocab_size = 32003  # not a multiple of 256

        # Case 1: max at the very first position
        logits = torch.zeros(1, vocab_size, device="cuda:0", dtype=torch.float32)
        logits[0, 0] = 100.0
        self._check_argmax(logits, vocab_size)

        # Case 2: max at the very last position
        logits = torch.zeros(1, vocab_size, device="cuda:0", dtype=torch.float32)
        logits[0, vocab_size - 1] = 100.0
        self._check_argmax(logits, vocab_size)

        # Case 3: max around the 256-boundary (thread boundary)
        for pos in [255, 256, 257, 511, 512, 513]:
            logits = torch.zeros(1, vocab_size, device="cuda:0", dtype=torch.float32)
            logits[0, pos] = 100.0
            self._check_argmax(logits, vocab_size)

        # Case 4: all identical values (argmax should be 0, the first occurrence)
        logits = torch.ones(1, vocab_size, device="cuda:0", dtype=torch.float32)
        layer = create_sampler_layer(vocab_size, temperature=0.0, seed=0)
        logits_efm = tensor_to_edge_fm_tensor(logits)
        token_ids_efm = tensor_to_edge_fm_tensor(
            torch.zeros(1, device="cuda:0", dtype=torch.int32)
        )
        layer.forward(logits_efm, token_ids_efm)
        torch.cuda.synchronize()
        result = torch.from_dlpack(token_ids_efm.to_dlpack())
        expected = torch.argmax(logits, dim=-1).to(torch.int32)
        assert torch.equal(result, expected), (
            f"All-identical edge case failed: got {result.item()}, expected {expected.item()}"
        )

        # Case 5: extreme values (inf, large negatives)
        logits = torch.full((1, vocab_size), -1e9, device="cuda:0", dtype=torch.float32)
        logits[0, 12345] = float('inf')
        self._check_argmax(logits, vocab_size)

    def _check_argmax(self, logits, vocab_size):
        """Helper: verify greedy path (topKStage1/2) matches torch.argmax."""
        batch_size = logits.shape[0]
        expected = torch.argmax(logits, dim=-1).to(torch.int32)
        layer = create_sampler_layer(vocab_size, temperature=0.0, seed=0)
        logits_efm = tensor_to_edge_fm_tensor(logits)
        token_ids_efm = tensor_to_edge_fm_tensor(
            torch.zeros(batch_size, device="cuda:0", dtype=torch.int32)
        )
        layer.forward(logits_efm, token_ids_efm)
        torch.cuda.synchronize()
        result = torch.from_dlpack(token_ids_efm.to_dlpack())
        assert torch.equal(result, expected), (
            f"Argmax mismatch: got {result}, expected {expected}"
        )

    @pytest.mark.order(1)
    @pytest.mark.parametrize("vocab_size", [32000, 151936])
    def test_greedy_vs_flashinfer_sampling(self, vocab_size):
        """对比 temperature=0 的 argmax 与 FlashInfer SamplingFromProb

        这个测试用来确认：FlashInfer SamplingFromProb 在 temperature 极小
        时的输出是否与正确的 argmax 一致。如果不一致，说明之前的对齐问题
        确实是 SamplingFromProb 在低温时的行为导致的。
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        batch_size = 4
        logits = torch.randn(batch_size, vocab_size, device="cuda:0", dtype=torch.float32)

        # Ground truth: torch.argmax
        expected = torch.argmax(logits, dim=-1).to(torch.int32)

        # EdgeFM greedy path (temperature=0, topKStage1/2)
        layer = create_sampler_layer(vocab_size, temperature=0.0, seed=42)
        logits_efm = tensor_to_edge_fm_tensor(logits.clone())
        token_ids_efm = tensor_to_edge_fm_tensor(
            torch.zeros(batch_size, device="cuda:0", dtype=torch.int32)
        )
        layer.forward(logits_efm, token_ids_efm)
        torch.cuda.synchronize()
        argmax_result = torch.from_dlpack(token_ids_efm.to_dlpack()).clone()

        # FlashInfer path: very small temperature (like 1e-9) through softmax + sampling
        tiny_temp = 1e-9
        logits_scaled = logits / tiny_temp
        probs = torch.softmax(logits_scaled, dim=-1)
        flashinfer_result = flashinfer.sampling.sampling_from_probs(
            probs, seed=42, offset=0
        ).to(torch.int32)

        argmax_matches = torch.equal(argmax_result, expected)
        flashinfer_matches = torch.equal(flashinfer_result, expected)

        print(f"\n[vocab_size={vocab_size}]")
        print(f"  torch.argmax (ground truth): {expected.tolist()}")
        print(f"  EdgeFM greedy (topK1/2):     {argmax_result.tolist()}  match={argmax_matches}")
        print(f"  FlashInfer SamplingFromProb:  {flashinfer_result.tolist()}  match={flashinfer_matches}")

        if not flashinfer_matches:
            mismatch_mask = flashinfer_result != expected
            n_mismatch = mismatch_mask.sum().item()
            print(f"  FlashInfer mismatches: {n_mismatch}/{batch_size}")
            for i in range(batch_size):
                if mismatch_mask[i]:
                    fi_tok = flashinfer_result[i].item()
                    exp_tok = expected[i].item()
                    print(f"    batch[{i}]: FlashInfer={fi_tok}, expected={exp_tok}, "
                          f"logit[fi]={logits[i, fi_tok].item():.6f}, "
                          f"logit[exp]={logits[i, exp_tok].item():.6f}")

        # Greedy path must always be correct
        assert argmax_matches, (
            f"EdgeFM greedy 结果与 torch.argmax 不一致！\n"
            f"  EdgeFM:  {argmax_result.tolist()}\n"
            f"  Expected: {expected.tolist()}"
        )

    @pytest.mark.order(1)
    def test_greedy_vs_sampling_on_close_logits(self):
        """验证 greedy 路径在 logits 差异极小时仍能正确选出最大值，
        而 OnlineSoftmax + SamplingFromProb 路径会因为概率采样而偏离 argmax。

        这复现了之前 generate 对齐失败的根本原因：当 temperature 虽然很小
        但未落入 argmax 分支（<1e-6）时，FlashInfer 的采样路径在 logits
        差异很小的情况下无法保证贪心行为。
        """
        torch.manual_seed(0)
        vocab = 151936
        batch = 16

        logits = torch.randn(batch, vocab, device="cuda:0", dtype=torch.float32) * 0.001
        for b in range(batch):
            peak = torch.randint(0, vocab, (1,)).item()
            logits[b, peak] += 0.002

        expected = torch.argmax(logits, dim=-1).to(torch.int32)

        # argmax path (temp=0): must be exact
        layer_greedy = create_sampler_layer(vocab, temperature=0.0, seed=42)
        l_efm = tensor_to_edge_fm_tensor(logits.clone())
        t_efm = tensor_to_edge_fm_tensor(torch.zeros(batch, device="cuda:0", dtype=torch.int32))
        layer_greedy.forward(l_efm, t_efm)
        torch.cuda.synchronize()
        greedy_result = torch.from_dlpack(t_efm.to_dlpack()).clone()

        assert torch.equal(greedy_result, expected), (
            f"Greedy path should be exact on close logits.\n"
            f"  got:      {greedy_result.tolist()}\n"
            f"  expected: {expected.tolist()}"
        )

        # sampling path (temp=1e-4, above the 1e-6 threshold): likely mismatches
        layer_sampling = create_sampler_layer(vocab, temperature=1e-4, seed=42)
        l_efm2 = tensor_to_edge_fm_tensor(logits.clone())
        t_efm2 = tensor_to_edge_fm_tensor(torch.zeros(batch, device="cuda:0", dtype=torch.int32))
        layer_sampling.forward(l_efm2, t_efm2)
        torch.cuda.synchronize()
        sampling_result = torch.from_dlpack(t_efm2.to_dlpack())

        n_mismatch = (sampling_result != expected).sum().item()
        print(f"\n  [close-logits] argmax exact=True, "
              f"sampling(temp=1e-4) mismatches={n_mismatch}/{batch}")

    @pytest.mark.order(1)
    def test_greedy_argmax_with_model_dump(self):
        """使用 Transformers 模型 decode dump 的真实 logits 验证 greedy 路径。

        decode_dump 包含 Transformers greedy decoding 每一步的 logits 和
        next_token_id。本测试将每步的 logits 送入 SamplerLayer(temperature=0)，
        验证 greedy (topKStage1/2) 输出与 Transformers 的 greedy token 完全一致。
        """
        dump_dir = Path(__file__).parent.parent / "data" / "decode_dump"
        manifest_path = dump_dir / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("decode_dump 数据不存在，跳过此测试")

        with open(manifest_path) as f:
            manifest = json.load(f)

        vocab_size = manifest["vocab_size"]
        num_steps = manifest["num_decode_steps"]
        ref_tokens = manifest["decode_tokens"]

        layer = create_sampler_layer(vocab_size, temperature=0.0, seed=0)

        # 1) prefill logits -> first decode token
        prefill_logits = np.load(dump_dir / "prefill_logits.npy")
        last_pos_logits = torch.from_numpy(
            prefill_logits[0, -1:, :].copy()
        ).to(device="cuda:0", dtype=torch.float32)
        token_out_efm = tensor_to_edge_fm_tensor(
            torch.zeros(1, device="cuda:0", dtype=torch.int32)
        )
        layer.forward(tensor_to_edge_fm_tensor(last_pos_logits), token_out_efm)
        torch.cuda.synchronize()
        prefill_token = torch.from_dlpack(token_out_efm.to_dlpack()).item()
        assert prefill_token == ref_tokens[0], (
            f"Prefill argmax mismatch: got {prefill_token}, expected {ref_tokens[0]}"
        )

        # 2) each decode step
        mismatches = []
        for step in range(num_steps):
            step_data = np.load(dump_dir / f"step_{step}.npz")
            logits_np = step_data["logits"]          # (vocab_size,)
            next_token_ref = int(step_data["next_token_id"])

            logits_gpu = torch.from_numpy(
                logits_np.reshape(1, -1).copy()
            ).to(device="cuda:0", dtype=torch.float32)
            token_out_efm = tensor_to_edge_fm_tensor(
                torch.zeros(1, device="cuda:0", dtype=torch.int32)
            )

            layer.forward(tensor_to_edge_fm_tensor(logits_gpu), token_out_efm)
            torch.cuda.synchronize()
            got = torch.from_dlpack(token_out_efm.to_dlpack()).item()

            if got != next_token_ref:
                mismatches.append((step, got, next_token_ref))

        assert len(mismatches) == 0, (
            f"EdgeFM greedy 与 Transformers greedy decoding 不一致 "
            f"({len(mismatches)}/{num_steps} steps):\n"
            + "\n".join(
                f"  step {s}: got={g}, expected={e}" for s, g, e in mismatches
            )
        )
        print(f"\n  [model-dump] {num_steps} decode steps + prefill: all argmax matched")

    @pytest.mark.order(1)
    def test_flashinfer_sampling_on_model_dump(self):
        """用同一份 dump logits 跑 FlashInfer SamplingFromProb，对比 argmax。

        展示 FlashInfer 的 OnlineSoftmax+SamplingFromProb 在真实模型 logits
        上、极小 temperature 时是否能等价于 greedy decoding。
        """
        dump_dir = Path(__file__).parent.parent / "data" / "decode_dump"
        manifest_path = dump_dir / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("decode_dump 数据不存在，跳过此测试")

        with open(manifest_path) as f:
            manifest = json.load(f)

        num_steps = manifest["num_decode_steps"]

        n_fi_mismatch = 0
        for step in range(num_steps):
            step_data = np.load(dump_dir / f"step_{step}.npz")
            logits_np = step_data["logits"]
            next_token_ref = int(step_data["next_token_id"])

            logits_gpu = torch.from_numpy(
                logits_np.reshape(1, -1).copy()
            ).to(device="cuda:0", dtype=torch.float32)

            # FlashInfer: OnlineSoftmax with tiny temperature, then sample
            tiny_temp = 1e-9
            probs = torch.softmax(logits_gpu / tiny_temp, dim=-1)
            fi_token = flashinfer.sampling.sampling_from_probs(
                probs, seed=42, offset=0
            ).item()

            if fi_token != next_token_ref:
                n_fi_mismatch += 1

        print(f"\n  [model-dump] FlashInfer SamplingFromProb(temp=1e-9) "
              f"mismatches: {n_fi_mismatch}/{num_steps}")

        # This test is informational: FlashInfer may or may not match.
        # The key insight is that greedy path (topKStage1/2) always matches (tested above).

    @pytest.mark.order(2)
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [32000, 50000])
    @pytest.mark.parametrize("temperature", [0.8, 1.0, 1.2])
    def test_forward_performance(self, batch_size, vocab_size, temperature):
        """测试采样前向传播的性能
        
        确保 EdgeFM 的性能不低于 FlashInfer 的 0.85 倍。
        使用中位数、更长测量时间、放宽阈值以降低微秒级 benchmark 的波动。
        
        参数:
            batch_size: 批次大小
            vocab_size: 词汇表大小
            temperature: 温度参数
        """
        # 设置随机种子
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # 创建输入 logits
        logits = torch.randn(batch_size, vocab_size, device="cuda:0", dtype=torch.float32)
        
        # 测试 FlashInfer 性能
        def run_flashinfer():
            logits_scaled = logits / temperature
            probs = torch.softmax(logits_scaled, dim=-1)
            generator = torch.Generator(device="cuda:0")
            generator.manual_seed(seed)
            flashinfer.sampling.sampling_from_probs(probs, deterministic=False, generator=generator)
        
        flashinfer_measurements = bench_gpu_time(run_flashinfer, repeat_time_ms=300)
        flashinfer_median = statistics.median(flashinfer_measurements)
        
        # 测试 edge_fm 性能
        layer = create_sampler_layer(vocab_size, temperature, seed=seed)
        
        logits_efm = tensor_to_edge_fm_tensor(logits)
        token_ids_efm = tensor_to_edge_fm_tensor(torch.zeros(batch_size, device="cuda:0", dtype=torch.int32))
        def run_edge_fm():
            layer.forward(logits_efm, token_ids_efm)
        
        edge_fm_measurements = bench_gpu_time(run_edge_fm, repeat_time_ms=300)
        edge_fm_median = statistics.median(edge_fm_measurements)
        
        # 计算性能比：FlashInfer 时间 / EdgeFM 时间
        performance_ratio = flashinfer_median / edge_fm_median if edge_fm_median > 0 else 0.0
        
        # 断言：EdgeFM 性能不低于 FlashInfer 的 0.85 倍（放宽阈值以降低微秒级测量的波动）
        assert performance_ratio >= 0.85, (
            f"性能测试失败：EdgeFM 性能低于 FlashInfer 的 0.85 倍\n"
            f"  batch_size={batch_size}, vocab_size={vocab_size}, temperature={temperature}\n"
            f"  FlashInfer 中位时间: {flashinfer_median:.4f} ms\n"
            f"  EdgeFM 中位时间: {edge_fm_median:.4f} ms\n"
            f"  性能比 (FlashInfer/EdgeFM): {performance_ratio:.4f}\n"
            f"  要求: performance_ratio >= 0.85"
        )


if __name__ == "__main__":
    # 支持直接运行：python test_sampler.py
    pytest.main([__file__, "-v"])
