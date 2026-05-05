from transformers import AutoModelForTokenClassification, BertTokenizerFast
from pathlib import Path
import json
import re
import torch


def extract_entities_from_outputs(tokenizer, input_ids, predictions, attention_mask, id2label):
    """从模型输出中提取实体（使用BIEO标签体系）
    只提取以B开头、以E结尾的完整实体
    参考format_outputs的逻辑简化实现
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    entities = []
    current_entity = []  # 存储当前实体的token列表
    flag = False  # 标记是否正在构建实体（遇到B后设为True）

    for i, (token, pred, mask) in enumerate(zip(tokens, predictions[0], attention_mask[0])):
        if mask.item() == 0:
            break  # padding位置，停止处理

        label_id = pred.item()
        label = id2label.get(label_id, 'O')

        # 跳过特殊token
        if token in ['[UNK]', '[CLS]', '[SEP]', '[PAD]']:
            if flag:
                # 如果正在构建实体，遇到特殊token则重置
                current_entity = []
                flag = False
            continue

        if label == 'B':
            # B标签：开始新实体
            if flag and current_entity:
                # 如果之前有未完成的实体（没有E结尾），丢弃
                current_entity = []
            current_entity = [token]
            flag = True
        elif label == 'I':
            # I标签：继续当前实体（只有flag=True时才接受）
            if flag:
                current_entity.append(token)
            else:
                # 异常：I标签前没有B，重置
                current_entity = []
                flag = False
        elif label == 'E':
            # E标签：结束当前实体
            if flag:
                current_entity.append(token)
                # 实体完整（B开头，E结尾），提取实体
                entity_text = _reconstruct_entity_text(current_entity)
                if entity_text:
                    entities.append(entity_text)
                current_entity = []
                flag = False
            else:
                # 异常：E标签前没有B，重置
                current_entity = []
                flag = False
        else:  # label == 'O'
            # O标签：结束当前实体（如果有未完成的实体，丢弃）
            if flag:
                current_entity = []
                flag = False

    # 处理最后一个实体（如果文本以实体结尾且以E结尾）
    if flag and current_entity:
        # 检查最后一个token的标签是否为E
        last_idx = len(current_entity) - 1
        # 需要找到最后一个token在predictions中的位置
        # 简化：如果current_entity不为空，说明可能以E结尾（在循环中已处理）
        # 实际上，如果flag还是True，说明没有遇到E，应该丢弃
        current_entity = []
        flag = False

    return list(set(entities))  # 去重


def _reconstruct_entity_text(entity_tokens):
    """从实体token列表重构实体文本
    entity_tokens: token字符串列表（不是元组列表）
    """
    if not entity_tokens:
        return ""

    entity_text = ""
    for token in entity_tokens:
        # 跳过[UNK]等特殊token
        if token in ['[UNK]', '[CLS]', '[SEP]', '[PAD]']:
            continue
        if token.startswith('##'):
            entity_text += token[2:]
        else:
            entity_text += token

    # 检查是否包含空格：如果包含空格，则不是有效实体
    if ' ' in entity_text or '\t' in entity_text:
        return ""

    entity_text = re.sub(r'\s+', '', entity_text)

    # 只返回长度>=2的实体
    if entity_text and len(entity_text) >= 2:
        return entity_text
    return ""


def run_medical_ner(tokenizer, model, device, id2label, sentence):
    """对单段文本做实体识别，返回实体字符串列表。"""
    sentence = (sentence or "").strip()
    if not sentence:
        return []
    sentence = sentence.replace(' ', '；').replace('\t', '；')
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(-1) * inputs["attention_mask"]
    entities = extract_entities_from_outputs(
        tokenizer,
        inputs["input_ids"].cpu(),
        predictions.cpu(),
        inputs["attention_mask"].cpu(),
        id2label,
    )
    return [e for e in entities if " " not in e and "\t" not in e]


if __name__ == '__main__':
    # 使用本地模型路径
    local_model_path = Path("E:/drug_kg/hub")

    # 查找模型目录（模型名称会被转换为目录格式）
    model_name = "iioSnail/bert-base-chinese-medical-ner"
    model_dir_name = model_name.replace('/', '--')
    model_path = local_model_path / f"models--{model_dir_name}"

    # 查找snapshots目录中的最新版本
    if model_path.exists():
        snapshots_dir = model_path / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                # 使用最新的snapshot
                actual_model_path = snapshots[-1]

                print(f"从本地加载模型: {actual_model_path}")
                # 强制使用本地文件，不使用缓存
                tokenizer = BertTokenizerFast.from_pretrained(
                    str(actual_model_path),
                    local_files_only=True  # 只使用本地文件，不查找缓存
                )
                model = AutoModelForTokenClassification.from_pretrained(
                    str(actual_model_path),
                    local_files_only=True  # 只使用本地文件，不查找缓存
                )
            else:
                raise FileNotFoundError(f"在 {model_path} 中未找到snapshots目录")
        else:
            # 如果直接是模型目录
            print(f"从本地加载模型: {model_path}")
            tokenizer = BertTokenizerFast.from_pretrained(
                str(model_path),
                local_files_only=True  # 只使用本地文件，不查找缓存
            )
            model = AutoModelForTokenClassification.from_pretrained(
                str(model_path),
                local_files_only=True  # 只使用本地文件，不查找缓存
            )
    else:
        raise FileNotFoundError(f"本地模型路径不存在: {model_path}")

    # 获取标签映射
    if hasattr(model.config, 'label2id') and model.config.label2id:
        label2id = model.config.label2id
        id2label = {v: k for k, v in label2id.items()}
    elif hasattr(model.config, 'id2label') and model.config.id2label:
        id2label = model.config.id2label
    else:
        id2label = {}

    # 手动选择设备：优先使用GPU（cuda），否则回退到CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)
    model.eval()

    # 项目根目录下的 cut/txt（本文件位于 drug_kg/NER/）
    cut_txt_dir = Path(__file__).resolve().parent.parent.parent / "cut" / "txt"
    if not cut_txt_dir.is_dir():
        raise FileNotFoundError(f"未找到目录: {cut_txt_dir}")

    json_paths = sorted(cut_txt_dir.glob("*.json"))
    result_list = []
    try:
        for json_path in json_paths:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
            for rec_idx, obj in enumerate(records):
                if not isinstance(obj, dict) or "text" not in obj:
                    continue
                sentence = obj["text"]
                filtered_entities = run_medical_ner(
                    tokenizer, model, device, id2label, sentence
                )
                row = {
                    "file": json_path.name,
                    "record_index": rec_idx,
                    "sentence": sentence,
                    "entities": filtered_entities,
                }
                if "chunk_index" in obj:
                    row["chunk_index"] = obj["chunk_index"]
                if "source" in obj:
                    row["source"] = obj["source"]
                result_list.append(row)

    except Exception as e:
        print(f"处理出错: {e}")
        import traceback

        traceback.print_exc()
    finally:
        out_path = Path(__file__).resolve().parent / "result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)
        print(
            f"{out_path} 已保存，共 {len(json_paths)} 个 JSON 文件，"
            f"{len(result_list)} 条含 text 的记录"
        )