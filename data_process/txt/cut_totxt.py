import json

chunks_dict = {}

data = json.load(open("../../content/final_data.jsonl", "r", encoding="utf-8"))
for item in data:
    chunk_index = item["text"]["corpus_chunk_index"]
    
    if chunk_index not in chunks_dict:
        chunks_dict[chunk_index] = []
    
    # 保存完整的信息
    chunks_dict[chunk_index].append({
        "text": item["text"]["text"],
        "source": item["text"]["source"],
        "chunk_index": item["text"]["corpus_chunk_index"]
    })

# 写入不同的 txt 文件（保存为JSON格式）
for idx, items in chunks_dict.items():
    filename = f"chunk_{idx}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"已生成: {filename}")