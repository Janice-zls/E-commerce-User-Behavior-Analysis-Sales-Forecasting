import os
import time

output_dir = r'g:\A.lsit\08.建模\校赛-B 题  销售商品行为预测\问题1'
files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
print(f"共 {len(files)} 张图表:\n")
for f in sorted(files):
    filepath = os.path.join(output_dir, f)
    mtime = os.path.getmtime(filepath)
    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    size = os.path.getsize(filepath) / 1024
    print(f"  {f}: {size:.1f} KB | 修改时间: {mtime_str}")
