#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的聚合模型进行推理
从JSON文件读取问题、上下文和分数，生成聚合结果
"""

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import argparse

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class AggregateInference:
    """聚合推理器"""

    def __init__(self, model_path: str):
        """初始化推理器"""
        self.device = torch.device("cuda:0")

        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": 0},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={0: "12GB"}
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"模型加载完成，设备: {next(self.model.parameters()).device}")

    def load_data(self, data_path: str) -> List[Dict]:
        """加载JSON数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("数据格式错误，期望JSON数组")

        print(f"加载了 {len(data)} 个样本")
        return data

    def extract_contexts_and_scores(self, item: Dict) -> tuple:
        """从数据项中提取上下文和分数"""
        # 提取context0-9
        contexts = []
        for i in range(10):
            context_key = f"context{i}"
            if context_key in item and item[context_key]:
                contexts.append(item[context_key].strip())

        # 提取外部分数
        scores = []
        max_para = item.get('max_para', 9)
        num_candidates = max_para + 1

        for i in range(num_candidates):
            key = f'external_similarity_avg{i}'
            if key in item:
                scores.append(item[key])
            else:
                scores.append(0.0)

        # 确保分数与上下文数量匹配
        scores = scores[:len(contexts)]

        return contexts, scores

    def select_top3_by_scores(self, contexts: List[str], scores: List[float]) -> tuple:
        """根据分数选择Top-3上下文"""
        if len(contexts) == 0:
            return [], []

        # 确保分数与上下文数量匹配
        scores = scores[:len(contexts)]

        # 按分数排序
        ranked_indices = list(np.argsort(scores)[::-1])
        top3_indices = ranked_indices[:min(3, len(ranked_indices))]
        top3_contexts = [contexts[i] for i in top3_indices]

        return top3_indices, top3_contexts

    def generate_aggregate(self, question: str, contexts: List[str],
                           top3_indices: List[int], top3_contexts: List[str],
                           max_new_tokens: int = 200) -> str:
        """生成聚合文本"""
        # 构建提示 - 只显示Top-3的context
        prompt_lines = [
            f"问题：{question}",
            "相关上下文："
        ]

        # 只显示Top-3的context，而不是全部
        for i, idx in enumerate(top3_indices, 1):
            if idx < len(contexts):
                prompt_lines.append(f"[{i}] {contexts[idx]}")

        prompt_lines.append("请给出聚合上下文：")
        prompt_text = "\n".join(prompt_lines)
        print(prompt_text)
        # 分词
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                do_sample=False,
                use_cache=False
            )

        # 解码
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取聚合部分
        split_marker = "请给出聚合上下文："
        pos = full_text.rfind(split_marker)
        if pos >= 0:
            aggregate_text = full_text[pos + len(split_marker):].strip()
        else:
            aggregate_text = full_text

        return aggregate_text

    def process_sample(self, item: Dict, max_new_tokens: int = 200) -> Dict:
        """处理单个样本"""
        question = item.get('question', '')
        sample_id = item.get('id', '')

        # 提取上下文和分数
        contexts, scores = self.extract_contexts_and_scores(item)

        # 选择Top-3
        top3_indices, top3_contexts = self.select_top3_by_scores(contexts, scores)

        # 生成聚合文本
        aggregate_text = self.generate_aggregate(
            question, contexts, top3_indices, top3_contexts, max_new_tokens
        )

        # 构建结果（确保所有数据类型可JSON序列化）
        result = {
            'id': sample_id,
            'question': question,
            'contexts': contexts,
            'scores': [float(s) for s in scores],  # 转换为Python float
            'top3_indices': [int(i) for i in top3_indices],  # 转换为Python int
            'top3_contexts': top3_contexts,
            'aggregate_text': aggregate_text,
            'simple_concatenation': '\n'.join(top3_contexts)  # 简单拼接作为对比
        }

        return result

    def run_inference(self, data_path: str, output_path: str, max_samples: int = None):
        """运行推理"""
        # 加载数据
        data = self.load_data(data_path)

        if max_samples:
            data = data[:max_samples]
            print(f"限制处理 {max_samples} 个样本")

        results = []

        print("开始推理...")
        for i, item in enumerate(data):
            try:
                result = self.process_sample(item)
                results.append(result)

                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1}/{len(data)} 个样本")

            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                continue

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"推理完成，结果保存到: {output_path}")
        print(f"成功处理 {len(results)} 个样本")

        return results


def main():
    parser = argparse.ArgumentParser(description='聚合推理')
    parser.add_argument('--model_path', type=str,
                        default="/hd/pengzewei/pro/redeep/ReDeEP/output/checkpoint_epoch_5_aggregate",
                        help='训练好的模型路径')
    parser.add_argument('--data_path', type=str,
                        default="./log/test_llama3_8B/llama3_8B_response_mu_1000.json",
                        help='数据文件路径')
    parser.add_argument('--output_path', type=str,
                        default="/hd/pengzewei/pro/redeep/ReDeEP/output//aggregate_inference_results_mu1000_3-8b.jsonl",
                        help='输出文件路径')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='最大处理样本数（用于测试）')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                        help='最大生成token数')

    args = parser.parse_args()

    # 初始化推理器
    inference = AggregateInference(args.model_path)

    # 运行推理
    results = inference.run_inference(
        args.data_path,
        args.output_path,
        args.max_samples
    )

    # 打印一些统计信息
    if results:
        print("\n=== 推理统计 ===")
        print(f"总样本数: {len(results)}")

        # 计算平均聚合文本长度
        avg_length = np.mean([len(r['aggregate_text']) for r in results])
        print(f"平均聚合文本长度: {avg_length:.1f} 字符")

        # 显示第一个样本的结果
        if len(results) > 0:
            sample = results[0]
            print(f"\n=== 样本示例 (ID: {sample['id']}) ===")
            print(f"问题: {sample['question']}")
            print(f"Top-3 索引: {sample['top3_indices']}")
            print(f"聚合文本: {sample['aggregate_text'][:200]}...")
            print(f"简单拼接: {sample['simple_concatenation'][:200]}...")


if __name__ == "__main__":
    main()
