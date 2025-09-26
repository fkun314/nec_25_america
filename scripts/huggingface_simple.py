from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

ckpt = "allenai/MolmoAct-7B-D-0812"

# ==== 可視化関数 ====
def visualize_trace_on_image(image, trace_coords, title="Visual Reasoning Trace"):
    """
    画像上にvisual reasoning traceを可視化する
    """
    # 座標を画像サイズに正規化
    actual_trace = normalize_coordinates_to_image(trace_coords, image)
    
    if len(actual_trace) > 0:
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

def normalize_coordinates_to_image(trace_coords, image):
    """
    0~255に正規化された座標を画像の実際のサイズに変換する
    """
    if not trace_coords or len(trace_coords) == 0:
        return []
    
    # ネストしたリスト構造を処理
    if isinstance(trace_coords[0], list) and len(trace_coords[0]) > 0 and isinstance(trace_coords[0][0], list):
        actual_trace = trace_coords[0]
    else:
        actual_trace = trace_coords
    
    # 画像のサイズを取得
    img_height, img_width = image.size[1], image.size[0]  # PIL Image: (width, height)
    
    # 座標を画像サイズに変換 (0~255 -> 0~image_size)
    normalized_trace = []
    for coord in actual_trace:
        x_norm = int((coord[0] / 255.0) * img_width)
        y_norm = int((coord[1] / 255.0) * img_height)
        normalized_trace.append([x_norm, y_norm])
    
    return normalized_trace

def analyze_trace_statistics(trace, image=None):
    """
    軌道の統計情報を分析する
    """
    if not trace or len(trace) == 0:
        return "No trajectory data available", []
    
    # ネストしたリスト構造を処理
    if isinstance(trace[0], list) and len(trace[0]) > 0 and isinstance(trace[0][0], list):
        # ネストしたリストの場合、最初の要素を取得
        actual_trace = trace[0]
    else:
        actual_trace = trace
    
    # 画像が提供されている場合は座標を正規化
    if image is not None:
        actual_trace = normalize_coordinates_to_image(trace, image)
    
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

# load the processor
processor = AutoProcessor.from_pretrained(
    ckpt,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map={"": "cuda:0"},
    padding_side="left",
)

# load the model
model = AutoModelForImageTextToText.from_pretrained(
    ckpt,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map={"": "cuda:0"},
)

# task instruction
instruction = "close the box"

prompt = (
    f"The task is {instruction}. "
    "What is the action that the robot should take. "
    f"To figure out the action that the robot should take to {instruction}, "
    "let's think through it step by step. "
    "First, what is the depth map for the first image? "
    "Second, what is the trajectory of the end effector in the first image? "
    "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
    "along with other images from different camera views as additional information, "
    "what is the action that the robot should take?"
)

# apply chat template
text = processor.apply_chat_template(
    [
        {
            "role": "user",
            "content": [dict(type="text", text=prompt)]
        }
    ], 
    tokenize=False, 
    add_generation_prompt=True,
)

# image observation (side + wrist)
url1 = "https://huggingface.co/allenai/MolmoAct-7B-D-0812/resolve/main/example_1.png"
url2 = "https://huggingface.co/allenai/MolmoAct-7B-D-0812/resolve/main/example_2.png"
r1 = requests.get(url1, headers={"User-Agent": "python-requests"}, timeout=30)
r1.raise_for_status()
r2 = requests.get(url2, headers={"User-Agent": "python-requests"}, timeout=30)
r2.raise_for_status()
img1 = Image.open(BytesIO(r1.content)).convert("RGB")
img2 = Image.open(BytesIO(r2.content)).convert("RGB")
imgs = [img1, img2]

# process the image and text
inputs = processor(
    images=[imgs],
    text=text,
    padding=True,
    return_tensors="pt",
)

# move inputs to the correct device
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# generate output
with torch.inference_mode():
    with torch.autocast("cuda:0", enabled=True, dtype=torch.bfloat16):
        generated_ids = model.generate(**inputs, max_new_tokens=256)

# only get generated tokens; decode them to text
generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print the generated text
print(f"generated text: {generated_text}")

# >>>  The depth map of the first image is ... The trajectory of the end effector in the first image is ...
#      Based on these information, along with other images from different camera views as additional information,
#      the action that the robot should take is ...

# parse out all depth perception tokens
depth = model.parse_depth(generated_text)
print(f"generated depth perception tokens: {depth}")

# >>>  [ "<DEPTH_START><DEPTH_1><DEPTH_2>...<DEPTH_END>" ]

# parse out all visual reasoning traces
trace = model.parse_trace(generated_text)
print(f"generated visual reasoning trace: {trace}")

# >>>  [ [[242, 115], [140, 77], [94, 58], [140, 44], [153, 26]]] ]

# parse out all actions, unnormalizing with key of "molmoact"
action = model.parse_action(generated_text, unnorm_key="molmoact")
print(f"generated action: {action}")

# >>>  [ [0.0732076061122558, 0.08228153779226191, -0.027760173818644346, 
#         0.15932856272248652, -0.09686601126895233, 0.043916773912953344, 
#         0.996078431372549] ]

# ==== 可視化処理 ====
print("\n=== Visual Reasoning Trace Analysis ===")
print("Trajectory coordinates: {}".format(trace))

# 統計情報を表示
if trace and len(trace) > 0:
    stats, actual_trace = analyze_trace_statistics(trace, img1)
    print("\n=== Trajectory Statistics (Normalized to Image Size) ===")
    for key, value in stats.items():
        print("{}: {}".format(key, value))
    
    # 元の座標（0~255）も表示
    print("\n=== Original Coordinates (0~255) ===")
    if isinstance(trace[0], list) and len(trace[0]) > 0 and isinstance(trace[0][0], list):
        original_trace = trace[0]
    else:
        original_trace = trace
    print("Original coordinates: {}".format(original_trace))
else:
    print("\n=== Trajectory Statistics ===")
    print("No trajectory data available")
    actual_trace = []

# 可視化
print("\nGenerating visualization images...")

# 各画像に軌道を重ねて表示
fig1 = visualize_trace_on_image(img1, trace, "Visual Reasoning Trace - {} Task - Image 1".format(instruction))
fig1.savefig('/home/hfujita/molmoact/scripts/simple_trace_visualization_img1.png', dpi=300, bbox_inches='tight')
print("Trajectory visualization saved: simple_trace_visualization_img1.png")

fig2 = visualize_trace_on_image(img2, trace, "Visual Reasoning Trace - {} Task - Image 2".format(instruction))
fig2.savefig('/home/hfujita/molmoact/scripts/simple_trace_visualization_img2.png', dpi=300, bbox_inches='tight')
print("Trajectory visualization saved: simple_trace_visualization_img2.png")

plt.show()
