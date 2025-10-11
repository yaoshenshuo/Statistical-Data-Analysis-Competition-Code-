import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler()
    ]
)

def download_file_with_progress(url, output_dir="downloads", max_retries=3):
    os.makedirs(output_dir, exist_ok=True)
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)
    
    # 检查文件是否已存在
    if os.path.exists(filepath):
        logging.info(f"文件已存在，跳过: {filename}")
        return True, filename
    
    for attempt in range(max_retries):
        try:
            logging.info(f"开始下载: {filename} (尝试 {attempt + 1})")
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # 创建进度条
            progress_bar = tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                desc=filename[:30],  # 限制描述长度
                leave=False
            )
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            logging.info(f"下载完成: {filename}")
            return True, filename
            
        except Exception as e:
            logging.error(f"下载失败 {filename}: {e}")
            time.sleep(3)
    
    logging.error(f"多次尝试后仍失败: {filename}")
    return False, filename

def smart_batch_download(url_file, output_dir="downloads", max_workers=8, batch_size=50):
    """
    智能批量下载，支持分批处理
    """
    # 读取URL
    with open(url_file, 'r', encoding='utf-8') as f:
        all_urls = [line.strip() for line in f if line.strip()]
    
    total_files = len(all_urls)
    logging.info(f"发现 {total_files} 个文件需要下载")
    
    # 分批处理，避免内存占用过大
    successful = []
    failed = []
    
    for i in range(0, total_files, batch_size):
        batch_urls = all_urls[i:i + batch_size]
        logging.info(f"处理批次 {i//batch_size + 1}: {len(batch_urls)} 个文件")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for url in batch_urls:
                futures.append(executor.submit(download_file_with_progress, url, output_dir))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理批次"):
                success, filename = future.result()
                if success:
                    successful.append(filename)
                else:
                    failed.append(filename)
        
        logging.info(f"当前批次完成，成功: {len(successful)}, 失败: {len(failed)}")
        time.sleep(1)  # 批次间短暂休息
    
    # 生成报告
    logging.info(f"\n最终结果:")
    logging.info(f"成功下载: {len(successful)} 个文件")
    logging.info(f"失败: {len(failed)} 个文件")
    
    if failed:
        logging.info("失败文件列表:")
        for file in failed:
            logging.info(f"  - {file}")
    
    return successful, failed

if __name__ == "__main__":
    URL_FILE = "2.txt"    # 你的URL文件
    OUTPUT_DIR = "downloads"      # 输出目录
    MAX_WORKERS = 10              # 并发数
    BATCH_SIZE = 100              # 每批处理数量
    
    successful, failed = smart_batch_download(URL_FILE, OUTPUT_DIR, MAX_WORKERS, BATCH_SIZE)