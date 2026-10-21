#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的Llama2-7b模型调用脚本 - 仅用于EM评估
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import numpy as np
import random
import re
from typing import List, Dict, Tuple

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class SimpleLlama2Chat:
    """简单的Llama2聊天模型"""

    def __init__(self, model_path: str = "/hd/pengzewei/pro/sunhao_dai/PLMs/llama2/shakechen/Llama-2-7b-chat-hf"):
        """
        初始化模型

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"正在加载模型: {model_path}")
        print(f"使用设备: {self.device}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": 0},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={0: "20GB"}
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ 模型加载完成")

    def normalize_text(self, text: str) -> str:
        """
        标准化文本，用于EM计算

        Args:
            text: 输入文本

        Returns:
            标准化后的文本
        """
        if not text:
            return ""

        # 转换为小写
        text = text.lower()

        # 移除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', '', text)

        # 移除多余空格
        text = ' '.join(text.split())

        return text.strip()

    def calculate_em_score(self, generated_text: str, reference_text: str) -> float:
        """
        计算Exact Match (EM) 分数

        Args:
            generated_text: 生成的文本
            reference_text: 参考文本（真实答案）

        Returns:
            EM分数 (0或1)
        """
        try:
            if not generated_text.strip() or not reference_text.strip():
                return 0.0

            # 标准化文本
            gen_normalized = self.normalize_text(generated_text)
            ref_normalized = self.normalize_text(reference_text)

            # 计算EM分数
            if gen_normalized == ref_normalized:
                return 1.0
            else:
                return 0.0

        except Exception as e:
            print(f"EM计算出错: {e}")
            return 0.0

    def chat_with_context(self, question: str, context: str, max_length: int = 30) -> str:
        """
        使用优化的阅读理解方法从上下文中提取答案

        Args:
            question: 问题
            context: 聚合的上下文
            max_length: 最大生成长度

        Returns:
            提取的答案
        """
        try:
            # 改进的阅读理解提示 - 更清晰、更具体
            input_text = f"""Based on the following context, answer the question with a concise answer (e.g., person name, place name, yes/no, number, etc.):

Context: {context}

Question: {question}

Answer:"""

            # 编码输入
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

            # 生成答案 - 使用更优化的参数
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.01,  # 降低温度以获得更确定的答案
                    do_sample=True,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.2,  # 增加重复惩罚
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3  # 避免重复n-gram
                )

            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取答案 - 改进的提取逻辑
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            elif "答案:" in generated_text:
                answer = generated_text.split("答案:")[-1].strip()
            else:
                # 尝试从输入文本之后提取
                answer = generated_text[len(input_text):].strip()

            # 清理答案 - 移除可能的重复前缀
            answer = answer.split("\n")[0].strip()  # 只取第一行
            answer = answer.split(".")[0].strip()  # 移除句号后的内容（如果答案应该很短）

            # 智能答案提取
            return self._extract_smart_answer(answer, question)

        except Exception as e:
            print(f"❌ 生成答案时出错: {e}")
            return "生成答案时出错"

    def _extract_smart_answer(self, answer: str, question: str) -> str:
        """
        智能提取答案，针对不同类型的问题使用不同的策略
        """
        answer = answer.strip()

        # 如果答案为空，直接返回
        if not answer:
            return "unknown"

        # 1. 处理yes/no问题 - 改进检测逻辑
        question_lower = question.lower()
        if any(word in question_lower for word in
               ['are', 'is', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'has', 'have',
                'had']):
            answer_lower = answer.lower()
            # 更精确的yes/no检测
            if answer_lower.startswith('yes') or answer_lower.startswith('correct'):
                return 'yes'
            elif answer_lower.startswith('no') or answer_lower.startswith('incorrect') or answer_lower.startswith(
                    'not'):
                return 'no'

        # 2. 处理who问题 - 提取完整人名
        if question_lower.startswith('who'):
            # 移除常见的描述性词汇，但保留完整人名
            stop_words = ['the', 'a', 'an', 'is', 'was', 'are', 'were', 'who', 'that', 'this', 'person', 'people']
            words = answer.split()
            meaningful_words = []
            for word in words:
                word_clean = word.strip('.,!?;:')
                if word_clean.lower() not in stop_words and len(word_clean) > 1:
                    meaningful_words.append(word_clean)

            if meaningful_words:
                # 限制长度，通常人名不会太长
                result = ' '.join(meaningful_words[:6])  # 最多6个词
                return result

        # 3. 处理where问题 - 提取地点信息
        if question_lower.startswith('where'):
            stop_words = ['the', 'a', 'an', 'is', 'was', 'are', 'were', 'where', 'that', 'this', 'location', 'place']
            words = answer.split()
            meaningful_words = []
            for word in words:
                word_clean = word.strip('.,!?;:')
                if word_clean.lower() not in stop_words and len(word_clean) > 1:
                    meaningful_words.append(word_clean)

            if meaningful_words:
                # 地点可能包含多个词（如 "New York City"）
                result = ' '.join(meaningful_words[:8])  # 最多8个词
                return result

        # 4. 处理what问题 - 提取完整信息
        if question_lower.startswith('what'):
            # 移除常见的描述性词汇，但保留完整信息
            stop_words = ['the', 'a', 'an', 'is', 'was', 'are', 'were', 'that', 'this', 'it', 'they', 'what']
            words = answer.split()
            meaningful_words = []
            for word in words:
                word_clean = word.strip('.,!?;:')
                if word_clean.lower() not in stop_words and len(word_clean) > 1:
                    meaningful_words.append(word_clean)

            if meaningful_words:
                # 限制长度，避免过长答案
                result = ' '.join(meaningful_words[:10])  # 最多10个词
                return result

        # 5. 处理数字问题 - 智能提取数字和单位
        if any(word in question_lower for word in ['how many', 'how much', 'number', 'count', 'amount']):
            # 先尝试提取数字+单位（改进正则表达式）
            number_with_unit = re.findall(r'\d+[,\d]*\.?\d*\s*\w+', answer)
            if number_with_unit:
                return number_with_unit[0].strip()

            # 如果没有单位，只提取数字（包括小数）
            numbers = re.findall(r'\d+[,\d]*\.?\d*', answer)
            if numbers:
                return numbers[0]

        # 6. 通用智能答案提取 - 优先选择简洁答案
        # 移除标点符号但保留重要信息
        answer_clean = answer.replace('!', '').replace('?', '').replace(':', '').replace(';', '')

        # 更宽松的停用词列表，保留更多信息
        stop_words = ['based', 'according', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of',
                      'and', 'or', 'but', 'so', 'yet', 'nor', 'for', 'as', 'if', 'when', 'where', 'why', 'how', 'what',
                      'who', 'which', 'that', 'this', 'these', 'those', 'please', 'unable', 'answer', 'given',
                      'passages', 'question', 'briefly', 'following', 'context', 'information', 'provided', 'text',
                      'says', 'states', 'mentions', 'accordingly']

        words = answer_clean.split()
        if words:
            # 跳过无意义的开头词
            while words and words[0].lower().strip('.,!?;:') in stop_words:
                words.pop(0)

            if words:
                # 智能提取：优先选择简洁答案
                meaningful_words = []
                for word in words:
                    word_clean = word.strip('.,!?;:')
                    # 保留所有有意义的词
                    if word_clean.lower() not in stop_words and len(word_clean) > 0:
                        meaningful_words.append(word_clean)
                    elif word_clean.lower() in ['yes', 'no']:  # 保留yes/no
                        meaningful_words.append(word_clean)

                if meaningful_words:
                    # 智能提取：优先选择完整答案
                    full_answer = ' '.join(meaningful_words)

                    # 如果答案长度合理，直接返回
                    if len(full_answer.split()) <= 8:
                        return full_answer
                    else:
                        # 如果答案过长，返回前几个词（但尝试保留完整短语）
                        # 优先返回前8个词，这通常足够表达答案
                        return ' '.join(meaningful_words[:8])

        # 如果所有处理都失败，返回原始答案
        return answer.strip() if answer.strip() else "unknown"

    def _smart_concatenate_contexts(self, top3_contexts: List[str], top3_indices: List[int],
                                    scores: List[float], all_contexts: List[str]) -> str:
        """
        智能拼接上下文 - 基于分数进行加权排序和去重

        Args:
            top3_contexts: Top-3上下文列表
            top3_indices: Top-3索引列表
            scores: 所有上下文的分数
            all_contexts: 所有上下文列表

        Returns:
            智能拼接后的上下文
        """
        if not top3_contexts:
            return ""

        # 如果有分数信息，按分数排序
        if scores and top3_indices and len(scores) > max(top3_indices):
            # 创建索引到分数的映射
            indexed_scores = [(top3_indices[i], scores[top3_indices[i]], top3_contexts[i])
                              for i in range(len(top3_contexts))]
            # 按分数降序排序
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            # 提取排序后的上下文
            sorted_contexts = [ctx for _, _, ctx in indexed_scores]
        else:
            sorted_contexts = top3_contexts

        # 去重 - 移除高度相似的上下文
        unique_contexts = []
        for ctx in sorted_contexts:
            is_duplicate = False
            for existing in unique_contexts:
                # 简单的相似度检查：如果两个上下文有超过80%的相同内容，认为是重复
                if self._context_similarity(ctx, existing) > 0.8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_contexts.append(ctx)

        # 用换行符连接，并添加分隔符以提高可读性
        if len(unique_contexts) > 1:
            return "\n\n".join([f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(unique_contexts)])
        else:
            return unique_contexts[0] if unique_contexts else ""

    def _context_similarity(self, ctx1: str, ctx2: str) -> float:
        """
        计算两个上下文的相似度（简单的基于词汇重叠的方法）

        Args:
            ctx1: 上下文1
            ctx2: 上下文2

        Returns:
            相似度分数 (0-1)
        """
        if not ctx1 or not ctx2:
            return 0.0

        # 转换为小写并分词
        words1 = set(ctx1.lower().split())
        words2 = set(ctx2.lower().split())

        if not words1 or not words2:
            return 0.0

        # 计算Jaccard相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def evaluate_with_aggregate_context(self,
                                        aggregate_results_path: str = "/hd/pengzewei/pro/redeep/ReDeEP/output/aggregate_inference_results_wk1000.jsonl",
                                        hotpot_data_path: str = "/hd/pengzewei/pro/redeep/ReDeEP/log/test_llama2_7B/dev.json",
                                        output_path: str = "./output/em_evaluation_resultswiki3.json",
                                        max_samples: int = 1000,
                                        use_support_facts: bool = False):
        """
        使用聚合上下文进行EM评估

        Args:
            aggregate_results_path: 聚合结果文件路径
            hotpot_data_path: Hotpot原始数据文件路径
            output_path: 输出结果文件路径
            max_samples: 最大评估样本数
            use_support_facts: 是否使用Hotpot原数据集中的support facts作为上下文
        """
        try:
            print("📊 开始EM评估...")
            print(f"聚合结果文件: {aggregate_results_path}")
            print(f"Hotpot数据文件: {hotpot_data_path}")

            # 加载聚合结果
            print("正在加载聚合结果...")
            aggregate_results = []
            with open(aggregate_results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        aggregate_results.append(json.loads(line))

            print(f"加载了 {len(aggregate_results)} 个聚合结果")

            if len(aggregate_results) == 0:
                print("❌ 聚合结果文件为空！")
                return None

            # 加载Hotpot原始数据
            print("正在加载Hotpot原始数据...")
            with open(hotpot_data_path, 'r', encoding='utf-8') as f:
                hotpot_data = json.load(f)

            print(f"加载了 {len(hotpot_data)} 个Hotpot样本")

            # 创建ID到原始数据的映射
            hotpot_dict = {item['_id']: item for item in hotpot_data}

            # 评估结果
            evaluation_results = []
            em_scores = []

            # 限制评估样本数
            actual_samples = min(len(aggregate_results), max_samples)
            eval_samples = aggregate_results[:actual_samples]
            print(f"📊 将评估 {actual_samples} 个样本 (聚合结果总数: {len(aggregate_results)}, 最大限制: {max_samples})")

            # for i, agg_result in enumerate(eval_samples):
            #     print(f"\n处理样本 {i + 1}/{len(eval_samples)}:")
            #     print(f"问题: {agg_result['question'][:100]}...")
            #
            #     # 获取上下文
            #     if use_support_facts:
            #         # 使用Hotpot原数据集中的support facts
            #         hotpot_id = agg_result.get('id', '')
            #         if hotpot_id in hotpot_dict:
            #             hotpot_item = hotpot_dict[hotpot_id]
            #             support_facts = hotpot_item.get('supporting_facts', [])
            #             context_list = hotpot_item.get('context', [])
            #
            #             # 正确提取support facts对应的实际句子内容
            #             context_sentences = []
            #             for fact in support_facts:
            #                 if len(fact) >= 2:
            #                     doc_name = fact[0]
            #                     sent_idx = fact[1]
            #
            #                     # 在context中找到对应的文档和句子
            #                     for doc in context_list:
            #                         if doc[0] == doc_name:  # 文档名匹配
            #                             if sent_idx < len(doc[1]):  # 句子索引有效
            #                                 sentence = doc[1][sent_idx]
            #                                 context_sentences.append(f"{doc_name}: {sentence}")
            #                             break
            #
            #             context = '\n'.join(context_sentences)
            #             print(f"使用support facts作为上下文，提取了 {len(context_sentences)} 个句子")
            #         else:
            #             print("⚠️ 未找到对应的Hotpot数据，跳过")
            #             continue
            #     else:
            #         # 使用聚合上下文
            #         context = agg_result.get('simple_concatenation', '')
            #         print(f"使用聚合上下文111")
            #
            #     if not context.strip():
            #         print("⚠️ 上下文为空，跳过")
            #         continue
            #
            #     # 生成答案
            #     generated_answer = self.chat_with_context(
            #         agg_result['question'],
            #         context
            #     )
            #
            #     print(f"生成答案: {generated_answer[:100]}...")
            #
            #     # 获取Hotpot原始答案
            #     hotpot_id = agg_result.get('id', '')
            #     if hotpot_id in hotpot_dict:
            #         hotpot_answer = hotpot_dict[hotpot_id].get('answer', '')
            #         print(f"原始答案: {hotpot_answer[:100]}...")
            #
            #         # 计算EM分数
            #         em_score = self.calculate_em_score(generated_answer, hotpot_answer)
            #         em_scores.append(em_score)
            #
            #         print(f"EM分数: {em_score}")
            #
            #         # 保存结果
            #         result_item = {
            #             'id': hotpot_id,
            #             'question': agg_result['question'],
            #             'context_used': context,
            #             'context_type': 'support_facts' if use_support_facts else 'aggregate',
            #             'generated_answer': generated_answer,
            #             'hotpot_answer': hotpot_answer,
            #             'em_score': em_score,
            #             'top3_indices': agg_result.get('top3_indices', []),
            #             'top3_contexts': agg_result.get('top3_contexts', [])
            #         }
            #         evaluation_results.append(result_item)
            #     else:
            #         print(f"⚠️ 未找到ID {hotpot_id} 对应的Hotpot数据")
            # 下面这个是使用所有context作为输入的context
            for i, agg_result in enumerate(eval_samples):
                print(f"\n处理样本 {i + 1}/{len(eval_samples)}:")
                print(f"问题: {agg_result['question'][:100]}...")

                # 获取上下
                if use_support_facts:  # 使用supportfact 上下文
                    # 使用Hotpot原数据集中的support facts
                    hotpot_id = agg_result.get('id', '')
                    if hotpot_id in hotpot_dict:
                        hotpot_item = hotpot_dict[hotpot_id]
                        support_facts = hotpot_item.get('supporting_facts', [])
                        context_list = hotpot_item.get('context', [])

                        # 提取support facts对应的完整段落
                        context_paragraphs = []
                        for fact in support_facts:
                            if len(fact) >= 2:
                                doc_name = fact[0]

                                # 在context中找到对应的文档，提取整个段落
                                for doc in context_list:
                                    if doc[0] == doc_name:  # 文档名匹配
                                        # 提取整个段落（所有句子）
                                        paragraph = ' '.join(doc[1])  # 将所有句子连接成段落
                                        context_paragraphs.append(f"{doc_name}: {paragraph}")
                                        break

                        context = '\n'.join(context_paragraphs)
                        print(f"使用support facts作为上下文，提取了 {len(context_paragraphs)} 个段落")
                    else:
                        print("⚠️ 未找到对应的Hotpot数据，跳过")
                        continue
                # else:
                #     # 随机挑选两个上下文
                #
                #     all_contexts = agg_result.get('contexts', [])
                #     if isinstance(all_contexts, str):
                #         all_contexts = [all_contexts] if all_contexts else []
                #     if all_contexts:
                #         sample_size = min(2, len(all_contexts))
                #         sampled_contexts = random.sample(all_contexts, sample_size)
                #         context = '\n'.join(sampled_contexts)
                #         print(f"使用聚合上下文（随机选取 {sample_size} 段）")
                #     else:
                #         context = agg_result.get('aggregate_text', '')
                #         print("上下文列表为空，回退使用聚合文本")
                else:
                    # 使用聚合上下文 或者 使用所有
                    # 优先使用模型生成的聚合文本，如果没有则使用智能拼接
                    aggregate_text = agg_result.get('xxx', '')
                    top3_contexts = agg_result.get('top3_contexts', [])
                    scores = agg_result.get('scores', [])
                    top3_indices = agg_result.get('top3_indices', [])

                    if aggregate_text and aggregate_text.strip():
                        # 使用模型生成的聚合文本（效果更好）
                        context = aggregate_text
                        print(f"使用模型生成的聚合文本")
                    elif top3_contexts and len(top3_contexts) > 0:
                        # 使用智能加权拼接（基于分数）
                        context = self._smart_concatenate_contexts(
                            top3_contexts, top3_indices, scores, agg_result.get('contexts', [])
                        )
                        print(f"使用智能加权拼接的上下文")
                    else:
                        # 回退到简单拼接
                        context = agg_result.get('simple_concatenation', '')
                        print(f"使用简单拼接的上下文")

                # if not context.strip():
                #     print("⚠️ 上下文为空，跳过")
                #     continue

                # 生成答案
                generated_answer = self.chat_with_context(
                    agg_result['question'],
                    context
                )

                print(f"生成答案: {generated_answer[:100]}...")

                # 获取Hotpot原始答案
                hotpot_id = agg_result.get('id', '')
                if hotpot_id in hotpot_dict:
                    hotpot_answer = hotpot_dict[hotpot_id].get('answer', '')
                    print(f"原始答案: {hotpot_answer[:100]}...")

                    # 计算EM分数
                    em_score = self.calculate_em_score(generated_answer, hotpot_answer)
                    em_scores.append(em_score)

                    print(f"EM分数: {em_score}")

                    # 保存结果
                    result_item = {
                        'id': hotpot_id,
                        'question': agg_result['question'],
                        'context_used': context,
                        'context_type': 'support_facts' if use_support_facts else 'aggregate',
                        'generated_answer': generated_answer,
                        'hotpot_answer': hotpot_answer,
                        'em_score': em_score,
                        'top3_indices': agg_result.get('top3_indices', []),
                        'top3_contexts': agg_result.get('top3_contexts', [])
                    }
                    evaluation_results.append(result_item)
                else:
                    print(f"⚠️ 未找到ID {hotpot_id} 对应的Hotpot数据")

            # 计算统计信息
            if em_scores:
                avg_em = np.mean(em_scores)
                max_em = np.max(em_scores)
                min_em = np.min(em_scores)
                std_em = np.std(em_scores)
                correct_count = sum(em_scores)

                print(f"\n📊 EM评估结果:")
                print(f"总样本数: {len(evaluation_results)}")
                print(f"正确样本数: {correct_count}")
                print(f"准确率: {correct_count / len(evaluation_results):.4f}")
                print(f"平均EM分数: {avg_em:.4f}")
                print(f"最高EM分数: {max_em:.4f}")
                print(f"最低EM分数: {min_em:.4f}")
                print(f"EM分数标准差: {std_em:.4f}")

                # 保存结果
                summary = {
                    'total_samples': len(evaluation_results),
                    'correct_samples': correct_count,
                    'accuracy': correct_count / len(evaluation_results),
                    'avg_em_score': avg_em,
                    'max_em_score': max_em,
                    'min_em_score': min_em,
                    'em_std': std_em,
                    'results': evaluation_results
                }
            else:
                summary = {
                    'total_samples': len(evaluation_results),
                    'correct_samples': 0,
                    'accuracy': 0.0,
                    'avg_em_score': 0.0,
                    'results': evaluation_results
                }

            # 保存结果到文件
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"\n✅ EM评估完成！结果已保存到: {output_path}")

            return summary

        except Exception as e:
            print(f"❌ EM评估出错: {e}")
            return {}


def main():
    """主函数 - 运行EM评估"""
    print("=" * 60)
    print("📊 EM评估 - 使用聚合上下文评估答案与Hotpot原答案的相似度")
    print("=" * 60)

    # 初始化模型
    try:
        chat_model = SimpleLlama2Chat()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 选择评估模式
    print("\n🔧 选择评估模式:")
    print("1. 使用聚合上下文 (默认)")
    print("2. 使用Hotpot原数据集中的support facts")

    try:
        choice = input("请输入选择 (1 或 2，直接回车使用默认): ").strip()
        use_support_facts = choice == "2"
    except:
        use_support_facts = False

    if use_support_facts:
        print("✅ 使用Hotpot原数据集中的support facts作为上下文")
    else:
        print("✅ 使用聚合上下文")

    # 检查聚合结果文件
    aggregate_path = "/hd/pengzewei/pro/redeep/ReDeEP/output/aggregate_inference_results.jsonl"
    try:
        with open(aggregate_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        sample_count = len([line for line in lines if line.strip()])
        print(f"📊 当前聚合结果文件包含 {sample_count} 个样本")

        if sample_count < 50:
            print("⚠️ 聚合结果样本数较少，建议生成更多样本")
            print("💡 提示：可以运行 inference_aggregate.py 生成更多聚合结果")
    except:
        print("⚠️ 无法读取聚合结果文件")

    # 运行EM评估
    results = chat_model.evaluate_with_aggregate_context(use_support_facts=use_support_facts)

    if results and 'results' in results:
        print("\n📋 评估结果摘要:")
        print("-" * 40)
        print(f"总样本数: {results['total_samples']}")
        print(f"正确样本数: {results['correct_samples']}")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"平均EM分数: {results['avg_em_score']:.4f}")

        # 显示前3个详细结果
        for i, result in enumerate(results['results'][:3]):
            print(f"\n样本 {i + 1}:")
            print(f"问题: {result['question']}")
            print(f"生成答案: {result['generated_answer'][:100]}...")
            print(f"原始答案: {result['hotpot_answer'][:100]}...")
            print(f"EM分数: {result['em_score']:.4f}")
    else:
        print("❌ EM评估失败，请检查数据文件")


if __name__ == "__main__":
    main()