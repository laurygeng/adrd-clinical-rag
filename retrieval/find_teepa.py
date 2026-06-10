import os
import shutil
from pathlib import Path

base_dir = "../knowledge_base"
source_dir = os.path.join(base_dir, "raw_files")
target_dir = os.path.join(base_dir, "vlm_visual_pdfs")
target_filename = "Communication_handouts_Teepa_snow.pdf"

def hunt_and_move():
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"🔍 开始在 {source_dir} 及其所有子文件夹中搜索...")
    
    found = False
    # rglob 会递归搜索所有子目录
    for filepath in Path(source_dir).rglob(target_filename):
        found = True
        print(f"🎯 找到了！它藏在这里: {filepath.parent}")
        
        dst_path = os.path.join(target_dir, target_filename)
        try:
            shutil.move(str(filepath), dst_path)
            print(f"✅ 成功将其抓取并移动到了: {target_dir}")
        except Exception as e:
            print(f"❌ 移动失败: {e}")
        break # 找到一个就停止
        
    if not found:
        print("⚠️ 奇怪，居然没找到。请确认它是否已经被你不小心删除了，或者名字被改动过。")

if __name__ == "__main__":
    hunt_and_move()