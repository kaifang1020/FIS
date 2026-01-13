import os
import re
import json
import math
import numpy as np

from swift.llm import PtEngine, RequestConfig, InferRequest

# ===== 0. 环境变量配置 =====
os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

# ===== 1. 路径配置 =====
# ① Qwen3-Omni 基座模型目录（你训练用的那个）
BASE_MODEL_DIR = "/gpfsnyu/scratch/km6704/qwen3_omni_30b_a3b"

# ② LoRA checkpoint 目录（注意：是具体某个 checkpoint，比如 checkpoint-100）
LORA_CKPT_DIR = (
    "/gpfsnyu/scratch/km6704/ondemand/jbnhandsome/train_result_with_thought_baseline/v0-20260110-155034/checkpoint-50"
)
# ③ 你刚才的评测集 json（就是你贴出来那种结构）
EVAL_JSON = (
    "/gpfsnyu/home/km6704/Mr_NT/dataset/fis_filtered_5008_training.json"
)

# ④ 评测结果输出到哪里
RESULTS_PATH = (
    "/gpfsnyu/home/km6704/mswift/ms-swift/fis_test_result_with_CoT_5008.jsonl"
)

# 🎯 设置起始索引：从第 1178 条开始（即跳过索引 0~1177）
START_INDEX = 596

# ===== 2. 工具函数 =====

def parse_score_from_text(gen_text: str):
    """
    修改后的逻辑：只从 </think> 标签之后的文本中提取第一个数字。
    """
    if not gen_text:
        return None

    # 寻找 </think> 标签，并截取其后的内容
    if "</think>" in gen_text:
        # split 后取最后一部分，防止中间出现干扰标签
        target_text = gen_text.split("</think>")[-1].strip()
    else:
        # 如果模型没有输出 </think> 标签，则回退到处理全文（或者根据需求返回 None）
        target_text = gen_text.strip()

    if not target_text:
        return None

    # 在目标文本中匹配第一个数字（支持正负号和小数）
    m = re.search(r"[-+]?\d+(\.\d+)?", target_text)
    if not m:
        return None

    try:
        return float(m.group(0))
    except Exception:
        return None


def extract_prompt_and_label(item):
    convs = item["conversations"]
    human = next(c for c in convs if c["from"] == "human")
    gpt   = next(c for c in convs if c["from"] == "gpt")

    prompt = human["value"]
    label_str = gpt["value"]

    try:
        label = float(label_str)
    except Exception:
        label = None

    return prompt, label


def compute_metrics(labels, preds):
    labels = np.array(labels, dtype=float)
    preds = np.array(preds, dtype=float)
    mae = np.mean(np.abs(labels - preds))
    rmse = math.sqrt(np.mean((labels - preds) ** 2))
    if len(labels) > 1 and np.std(labels) > 0 and np.std(preds) > 0:
        corr = np.corrcoef(labels, preds)[0, 1]
    else:
        corr = float("nan")
    return mae, rmse, corr


# ===== 3. 主程序 =====

def main():
    print("==== FIS LoRA 多模态评测（续跑模式） ====")
    print(f"起始索引 : {START_INDEX}")
    print(f"结果路径 : {RESULTS_PATH}")

    # --- [Step 1] 加载已有的结果，用于同步全量指标 ---
    all_labels = []
    all_preds = []
    if os.path.exists(RESULTS_PATH):
        print(f"正在读取已有文件以加载前 {START_INDEX} 条的指标...")
        with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get('pred') is not None:
                            all_labels.append(data['label'])
                            all_preds.append(data['pred'])
                    except:
                        continue
        print(f"已加载 {len(all_preds)} 条有效历史预测。")

    # --- [Step 2] 加载推理引擎 ---
    print("\n[Step 2] 加载 PtEngine + LoRA ...")
    engine = PtEngine(
        BASE_MODEL_DIR,
        adapters=[LORA_CKPT_DIR],
        max_batch_size=1,
    )

    request_config = RequestConfig(
        max_tokens=1024,
        temperature=0.0,
    )

    print("\n[Step 3] 加载 eval 数据集...")
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"数据集总规模: {len(dataset)}")

    # 🎯 使用 "a" 模式打开文件，新结果将追加在末尾
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    fw = open(RESULTS_PATH, "a", encoding="utf-8")

    print(f"\n[Step 4] 开始推理（从 idx={START_INDEX} 开始）...\n")
    
    for idx, item in enumerate(dataset):
        # 🎯 跳过逻辑
        if idx < START_INDEX:
            continue

        video_path = item.get("video", "")
        prompt, label = extract_prompt_and_label(item)

        if label is None:
            print(f"[WARN] idx={idx} label 解析失败，跳过。")
            continue

        videos_arg = [video_path] if os.path.exists(video_path) else None

        infer_req = InferRequest(
            messages=[{"role": "user", "content": prompt}],
            videos=videos_arg,
        )

        try:
            resp_list = engine.infer([infer_req], request_config)
            gen_text = resp_list[0].choices[0].message.content
        except Exception as e:
            print(f"[ERROR] idx={idx} 推理报错：{repr(e)}")
            gen_text = ""
        
        # 🎯 提取 </think> 之后的分数
        score = parse_score_from_text(gen_text)
        print(f"[{idx+1}/{len(dataset)}] Score: {score} | GT: {label}")

        record = {
            "idx": idx,
            "video": video_path,
            "label": label,
            "raw_text": gen_text,
            "pred": score,
        }
        fw.write(json.dumps(record, ensure_ascii=False) + "\n")
        fw.flush()

        if score is not None:
            all_labels.append(label)
            all_preds.append(score)

    fw.close()
    print(f"\n评测完成，结果已追加至：{RESULTS_PATH}")

    # ===== 5. 计算最终全量指标 =====
    if len(all_labels) > 0:
        mae, rmse, corr = compute_metrics(all_labels, all_preds)
        print("\n====== FIS 全量汇总指标（历史 + 新跑） ======")
        print(f"有效样本总数: {len(all_labels)}")
        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Pearson corr: {corr:.4f}")

if __name__ == "__main__":
    main()