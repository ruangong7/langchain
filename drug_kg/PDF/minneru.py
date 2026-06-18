import os
import json
import time
from mineru import MinerU
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

# ================== 请用新 Token 替换以下值（当前 Token 已泄露） ==================
TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MzAwNzEzMyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc4MTY4NzY1NiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTc4NTM0NjY1MjgiLCJvcGVuSWQiOm51bGwsInV1aWQiOiIwOTBmZjA1YS04NDQzLTRjYzQtYTY3Yi00NTE4ZTFlMzU2MzEiLCJlbWFpbCI6IiIsImV4cCI6MTc4OTQ2MzY1Nn0.sqkXRecb6yvndA71IfNQR_phqwE4tVOtqIc-dRxgDpafhAj5LRug81JOmJk0qQbzMljU9bRuI05wFiZIDcTSCw"
PDF_DIR = r"F:\langchain_legacy\docs"
OUTPUT_DIR = r"F:\langchain_legacy\drug_kg\PDF\outputs_mineru"
# ========================================================================

# ---------- 带重试的批量提交函数 ----------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    reraise=True
)
def safe_extract_batch(client, batch):
    return client.extract_batch(batch)

# ---------- 初始化客户端 ----------
client = MinerU(TOKEN)

# ---------- 收集所有 PDF ----------
pdf_files = []
for root, dirs, files in os.walk(PDF_DIR):
    for file in files:
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(root, file))

print(f"✅ 共发现 {len(pdf_files)} 个PDF文件，开始批量解析...")

# ---------- 配置 ----------
BATCH_SIZE = 5          # 每次提交 5 个文件（保守值）
REQUEST_INTERVAL = 5    # 每批间隔 5 秒
total_batches = (len(pdf_files) + BATCH_SIZE - 1) // BATCH_SIZE

# ---------- 主循环 ----------
for i in range(0, len(pdf_files), BATCH_SIZE):
    batch = pdf_files[i:i+BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    print(f"\n⏳ [批次 {batch_num}/{total_batches}] 处理第 {i+1}~{min(i+BATCH_SIZE, len(pdf_files))} 个文件...")
    
    try:
        for result in safe_extract_batch(client, batch):
            save_dir = os.path.join(OUTPUT_DIR, os.path.splitext(result.filename)[0])
            os.makedirs(save_dir, exist_ok=True)
            
            # 只保存 JSON（使用 content_list）
            json_path = os.path.join(save_dir, "result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result.content_list, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ 已保存 JSON: {result.filename}")
    except Exception as e:
        print(f"❌ 批次 {batch_num} 彻底失败，跳过: {e}")
        # 继续下一批

    # 批次间延迟（最后一批不等待）
    if i + BATCH_SIZE < len(pdf_files):
        print(f"⏸️  等待 {REQUEST_INTERVAL} 秒后继续...")
        time.sleep(REQUEST_INTERVAL)

print("\n🎉 全部处理完成！")