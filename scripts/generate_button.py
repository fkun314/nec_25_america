from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

device = "cuda:0"
ckpt = "allenai/MolmoAct-7B-D-0812"

# GPUメモリをクリア
torch.cuda.empty_cache()

# load
processor = AutoProcessor.from_pretrained(
    ckpt,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # より軽量なデータ型を使用
    device_map={"": device},
    padding_side="left",
)

model = AutoModelForImageTextToText.from_pretrained(
    ckpt,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # より軽量なデータ型を使用
    device_map={"": device},
    low_cpu_mem_usage=True,  # CPUメモリ使用量を削減
)

# ==== 画像リサイズ関数 ====
def resize_image(image, max_size=512):
    """画像を指定した最大サイズにリサイズする"""
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# ==== 可視化関数 ====
def visualize_trace_on_image(image, trace_coords, title="Visual Reasoning Trace"):
    """
    画像上にvisual reasoning traceを可視化する
    """
    # ネストしたリスト構造を処理
    if trace_coords and len(trace_coords) > 0:
        if isinstance(trace_coords[0], list) and len(trace_coords[0]) > 0 and isinstance(trace_coords[0][0], list):
            # ネストしたリストの場合、最初の要素を取得
            actual_trace = trace_coords[0]
        else:
            actual_trace = trace_coords
        
        coords = np.array(actual_trace)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
    else:
        x_coords = []
        y_coords = []
    
    # プロットを作成
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 画像を表示
    ax.imshow(image)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # 軌道をプロット
    if len(x_coords) > 0:
        # 軌道の線を描画
        ax.plot(x_coords, y_coords, 'r-', linewidth=3, alpha=0.8, label='Trajectory')
        
        # 各点をマーク
        ax.scatter(x_coords, y_coords, c='red', s=100, alpha=0.9, 
                  edgecolors='white', linewidth=2, label='Trajectory Points')
        
        # 開始点と終了点を特別にマーク
        if len(x_coords) > 0:
            ax.scatter(x_coords[0], y_coords[0], c='green', s=150, 
                      marker='o', edgecolors='white', linewidth=2, label='Start Point')
        if len(x_coords) > 1:
            ax.scatter(x_coords[-1], y_coords[-1], c='blue', s=150, 
                      marker='s', edgecolors='white', linewidth=2, label='End Point')
        
        # 座標をテキストで表示
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax.annotate('({},{})'.format(x, y), (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No trajectory data available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    return fig

def analyze_trace_statistics(trace):
    """
    軌道の統計情報を分析する
    """
    if not trace or len(trace) == 0:
        return "No trajectory data available"
    
    # ネストしたリスト構造を処理
    if isinstance(trace[0], list) and len(trace[0]) > 0 and isinstance(trace[0][0], list):
        # ネストしたリストの場合、最初の要素を取得
        actual_trace = trace[0]
    else:
        actual_trace = trace
    
    coords = np.array(actual_trace)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # 距離を計算
    distances = []
    for i in range(1, len(coords)):
        dist = np.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)
        distances.append(dist)
    
    total_distance = sum(distances)
    
    stats = {
        "Number of Points": len(actual_trace),
        "Start Coordinates": [int(x_coords[0]), int(y_coords[0])],
        "End Coordinates": [int(x_coords[-1]), int(y_coords[-1])],
        "Total Distance": round(total_distance, 2),
        "Average Distance": round(np.mean(distances) if distances else 0, 2),
        "X Coordinate Range": [int(np.min(x_coords)), int(np.max(x_coords))],
        "Y Coordinate Range": [int(np.min(y_coords)), int(np.max(y_coords))]
    }
    
    return stats, actual_trace

# ==== 写真を3枚ロードしてリサイズ ====
img_wide1 = Image.open("scripts/img/IMG_7522.png").convert("RGB")
img_wide1 = resize_image(img_wide1, max_size=512)

img_wide2 = Image.open("scripts/img/IMG_7523.png").convert("RGB")
img_wide2 = resize_image(img_wide2, max_size=512)

img_wrist = Image.open("scripts/img/IMG_7524.png").convert("RGB")
img_wrist = resize_image(img_wrist, max_size=512)

print(f"Resized image sizes: {img_wide1.size}, {img_wide2.size}, {img_wrist.size}")

# ==== 英語プロンプト ====
instruction = "press the red button"
prompt = (
    f"The task is {instruction}. "
    "What is the action that the robot should take. "
    f"To figure out the action that the robot should take to {instruction}, let's think through it step by step. "
    "First, what is the depth map for this image? "
    "Second, what is the trajectory of the end effector? "
    "Based on the depth map of the image and the trajectory of the end effector, what is the action that the robot should take?"
)

# チャットテンプレートを適用
messages = [
    {
        "role": "user",
        "content": [dict(type="text", text=prompt)]
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)

# ==== processorでまとめる ====
inputs = processor(
    images=[[img_wide1, img_wide2, img_wrist]],  # ネストしたリスト形式
    text=text,
    padding=True,
    return_tensors="pt"
).to(device)

# ==== 生成 ====
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=processor.tokenizer.eos_token_id
    )

generated_tokens = output[:, inputs["input_ids"].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

print("***** model prediction *****")
print(generated_text)

# ==== optional: 行動ベクトル解析 ====
# depth = model.parse_depth(generated_text)
trace = model.parse_trace(generated_text)
action = model.parse_action(generated_text, unnorm_key="molmoact")

print("\n***** Analysis Results *****")
print(f"Trajectory coordinates: {trace}")
print(f"Action: {action}")

# ==== 可視化処理 ====
print("\n=== Visual Reasoning Trace Analysis ===")
print("Trajectory coordinates: {}".format(trace))

# 統計情報を表示
stats, actual_trace = analyze_trace_statistics(trace)
print("\n=== Trajectory Statistics ===")
for key, value in stats.items():
    print("{}: {}".format(key, value))

# 可視化
print("\nGenerating visualization images...")

# 最初の画像（wide1）に軌道を重ねて表示
fig1 = visualize_trace_on_image(img_wide1, trace, "Visual Reasoning Trace - Press Red Button Task")
fig1.savefig('/home/hfujita/molmoact/scripts/trace_visualization_wide1.png', dpi=300, bbox_inches='tight')
print("Trajectory visualization saved: trace_visualization_wide1.png")

# 2つ目の画像（wide2）にも表示
fig2 = visualize_trace_on_image(img_wide2, trace, "Visual Reasoning Trace - Second Camera View")
fig2.savefig('/home/hfujita/molmoact/scripts/trace_visualization_wide2.png', dpi=300, bbox_inches='tight')
print("Trajectory visualization saved: trace_visualization_wide2.png")

# 3つ目の画像（wrist）にも表示
fig3 = visualize_trace_on_image(img_wrist, trace, "Visual Reasoning Trace - Wrist Camera View")
fig3.savefig('/home/hfujita/molmoact/scripts/trace_visualization_wrist.png', dpi=300, bbox_inches='tight')
print("Trajectory visualization saved: trace_visualization_wrist.png")

plt.show()
