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
instruction = "catch the game controller"

# より具体的で詳細なプロンプト

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
# → このコードは使わないです。
# prompt = (
#     f"The task is to {instruction}. "
#     "This is a robotic manipulation task where the robot needs to grasp a game controller. "
#     "Let's analyze this step by step:\n\n"
#     "1. First, identify the game controller in the image. Look for rectangular objects that could be controllers.\n"
#     "2. Analyze the depth map to understand the 3D structure and distance to the controller.\n"
#     "3. Plan a trajectory for the end effector to approach the controller from above or from the side.\n"
#     "4. The trajectory should be smooth and avoid obstacles.\n"
#     "5. The final action should position the gripper to grasp the controller firmly.\n\n"
#     "Based on the depth map and visual reasoning trace, what is the precise action the robot should take to successfully grasp the game controller?"
# )

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

img1 = Image.open("scripts/img/IMG_7527.png").convert("RGB")
img2 = Image.open("scripts/img/IMG_7529.png").convert("RGB")
imgs = [img1, img2]

# process the image and text
inputs = processor(
    images=[imgs],
    text=text,
    padding=True,
    return_tensors="pt",
)

# move inputs to cuda:0 (モデルとプロセッサと同じデバイス)
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

# 安全な生成を実行する関数
def generate_safe_output(model, inputs, generation_params=None):
    """
    安全な生成を実行し、エラーを適切に処理する
    """
    if generation_params is None:
        generation_params = {
            'max_new_tokens': 256,
            'do_sample': False,  # デフォルトは貪欲デコーディング
            'pad_token_id': processor.tokenizer.eos_token_id,
        }
    
    try:
        with torch.inference_mode():
            with torch.autocast("cuda:0", enabled=True, dtype=torch.bfloat16):
                generated_ids = model.generate(**inputs, **generation_params)

        # 生成されたテキストを取得
        generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # 軌道を解析
        trace = model.parse_trace(generated_text)
        action = model.parse_action(generated_text, unnorm_key="molmoact")
        
        return {
            'text': generated_text,
            'trace': trace,
            'action': action,
            'success': True,
            'params': generation_params
        }
        
    except Exception as e:
        print(f"Generation failed with params {generation_params}: {str(e)}")
        return {
            'text': "",
            'trace': [],
            'action': [],
            'success': False,
            'params': generation_params,
            'error': str(e)
        }

# 複数回の生成を実行して最良の結果を選択
def generate_multiple_outputs(model, inputs, num_generations=3):
    """
    複数回の生成を実行し、最良の結果を選択する
    """
    all_results = []
    
    # 異なる生成パラメータのセットを定義
    generation_configs = [
        # 設定1: 貪欲デコーディング（最も安定）
        {
            'max_new_tokens': 256,
            'do_sample': False,
            'pad_token_id': processor.tokenizer.eos_token_id,
        },
        # 設定2: サンプリング（低いtemperature）
        {
            'max_new_tokens': 256,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'pad_token_id': processor.tokenizer.eos_token_id,
        },
        # 設定3: サンプリング（中程度のtemperature）
        {
            'max_new_tokens': 256,
            'do_sample': True,
            'temperature': 0.8,
            'top_p': 0.85,
            'pad_token_id': processor.tokenizer.eos_token_id,
        }
    ]
    
    for i in range(min(num_generations, len(generation_configs))):
        print(f"Generating output {i+1}/{num_generations}...")
        
        result = generate_safe_output(model, inputs, generation_configs[i])
        all_results.append(result)
        
        if result['success']:
            trace = result['trace']
            trace_length = len(trace[0]) if trace and trace[0] else 0
            print(f"Generation {i+1} - Success: {trace_length} trace points")
        else:
            print(f"Generation {i+1} - Failed: {result.get('error', 'Unknown error')}")
    
    return all_results

# 複数回の生成を実行
print("Starting multiple generation process...")
generation_results = generate_multiple_outputs(model, inputs, num_generations=3)

# 最良の結果を選択（成功した生成の中から）
best_result = None
best_score = -1

# 成功した生成結果のみを対象とする
successful_results = [r for r in generation_results if r['success']]

if not successful_results:
    print("\nNo successful generations found. Using fallback generation...")
    # フォールバック: 最もシンプルな設定で再試行
    fallback_params = {
        'max_new_tokens': 128,
        'do_sample': False,
        'pad_token_id': processor.tokenizer.eos_token_id,
    }
    fallback_result = generate_safe_output(model, inputs, fallback_params)
    if fallback_result['success']:
        best_result = fallback_result
    else:
        print("Fallback generation also failed. Please check model and input configuration.")
        exit(1)
else:
    for i, result in enumerate(successful_results):
        trace = result['trace']
        if trace and len(trace) > 0 and len(trace[0]) > 0:
            # 軌道の長さと多様性をスコアリング
            actual_trace = trace[0] if isinstance(trace[0][0], list) else trace
            coords = np.array(actual_trace)
            
            # 軌道の長さ
            trajectory_length = len(actual_trace)
            
            # 軌道の多様性（座標の分散）
            if len(coords) > 1:
                x_var = np.var(coords[:, 0])
                y_var = np.var(coords[:, 1])
                diversity_score = x_var + y_var
            else:
                diversity_score = 0
            
            # 総合スコア
            score = trajectory_length * 0.7 + diversity_score * 0.3
            
            print(f"Generation {i+1} score: {score:.2f} (length: {trajectory_length}, diversity: {diversity_score:.2f})")
            
            if score > best_score:
                best_score = score
                best_result = result

if best_result:
    print(f"\nBest result selected: Generation with score {best_score:.2f}")
    generated_text = best_result['text']
    trace = best_result['trace']
    action = best_result['action']
else:
    print("\nNo valid results found, using first successful generation")
    generated_text = successful_results[0]['text']
    trace = successful_results[0]['trace']
    action = successful_results[0]['action']

# 結果を表示
print(f"\n=== Best Generation Result ===")
print(f"Generated text: {generated_text}")

# 深度情報を解析
depth = model.parse_depth(generated_text)
print(f"Generated depth perception tokens: {depth}")

# 軌道とアクションを表示
print(f"Generated visual reasoning trace: {trace}")
print(f"Generated action: {action}")

# 全生成結果の比較表示
print(f"\n=== All Generation Results Comparison ===")
for i, result in enumerate(generation_results):
    if result['success']:
        trace = result['trace']
        trace_length = len(trace[0]) if trace and trace[0] else 0
        params = result['params']
        temp = params.get('temperature', 'N/A')
        top_p = params.get('top_p', 'N/A')
        print(f"Generation {i+1}: {trace_length} trace points, temp={temp}, top_p={top_p}")
    else:
        print(f"Generation {i+1}: FAILED - {result.get('error', 'Unknown error')}")

# ==== 改善された可視化処理 ====
print("\n=== Enhanced Visual Reasoning Trace Analysis ===")
print("Best trajectory coordinates: {}".format(trace))

# 統計情報を表示
if trace and len(trace) > 0:
    stats, actual_trace = analyze_trace_statistics(trace, img1)
    print("\n=== Best Trajectory Statistics (Normalized to Image Size) ===")
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
    print("\n=== Best Trajectory Statistics ===")
    print("No trajectory data available")
    actual_trace = []

# 全生成結果の軌道を比較可視化
def visualize_all_traces(images, generation_results, instruction):
    """
    全生成結果の軌道を比較表示する
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(generation_results):
        if i >= 6:  # 最大6つの結果まで表示
            break
            
        trace = result['trace']
        ax = axes[i]
        
        # 画像を表示
        ax.imshow(images[0])  # 最初の画像を使用
        
        # 軌道を描画
        if trace and len(trace) > 0:
            # 座標を画像サイズに正規化
            actual_trace = normalize_coordinates_to_image(trace, images[0])
            
            if len(actual_trace) > 0:
                coords = np.array(actual_trace)
                x_coords = coords[:, 0]
                y_coords = coords[:, 1]
                
                # 軌道の線を描画
                ax.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.8)
                ax.scatter(x_coords, y_coords, c='red', s=50, alpha=0.9)
                
                # 開始点と終了点をマーク
                if len(x_coords) > 0:
                    ax.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Start')
                if len(x_coords) > 1:
                    ax.scatter(x_coords[-1], y_coords[-1], c='blue', s=100, marker='s', label='End')
        
        params = result.get('params', {})
        temp = params.get('temperature', 'N/A')
        top_p = params.get('top_p', 'N/A')
        ax.set_title(f"Generation {i+1}\nTemp: {temp}, Top-p: {top_p}\nPoints: {len(actual_trace) if 'actual_trace' in locals() else 0}", 
                    fontsize=10)
        ax.axis('off')
    
    # 残りのサブプロットを非表示
    for i in range(len(generation_results), 6):
        axes[i].axis('off')
    
    plt.suptitle(f"All Generation Results Comparison - {instruction} Task", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# 可視化
print("\nGenerating enhanced visualization images...")

# 全生成結果の比較可視化
comparison_fig = visualize_all_traces(imgs, generation_results, instruction)
comparison_fig.savefig('/home/hfujita/molmoact/scripts/huggingface_all_generations_comparison.png', dpi=300, bbox_inches='tight')
print("All generations comparison saved: huggingface_all_generations_comparison.png")

# 最良の結果の詳細可視化
fig1 = visualize_trace_on_image(img1, trace, "Best Visual Reasoning Trace - {} Task - Image 1".format(instruction))
fig1.savefig('/home/hfujita/molmoact/scripts/huggingface_trace_visualization_img1.png', dpi=300, bbox_inches='tight')
print("Best trajectory visualization saved: huggingface_trace_visualization_img1.png")

fig2 = visualize_trace_on_image(img2, trace, "Best Visual Reasoning Trace - {} Task - Image 2".format(instruction))
fig2.savefig('/home/hfujita/molmoact/scripts/huggingface_trace_visualization_img2.png', dpi=300, bbox_inches='tight')
print("Best trajectory visualization saved: huggingface_trace_visualization_img2.png")

# 軌道の品質評価
def evaluate_trajectory_quality(trace, image=None):
    """
    軌道の品質を評価する
    """
    if not trace or len(trace) == 0:
        return {"quality_score": 0, "issues": ["No trajectory data"]}
    
    if isinstance(trace[0], list) and len(trace[0]) > 0 and isinstance(trace[0][0], list):
        actual_trace = trace[0]
    else:
        actual_trace = trace
    
    # 画像が提供されている場合は座標を正規化
    if image is not None:
        actual_trace = normalize_coordinates_to_image(trace, image)
    
    if len(actual_trace) < 2:
        return {"quality_score": 0, "issues": ["Insufficient trajectory points"]}
    
    coords = np.array(actual_trace)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    issues = []
    score = 100
    
    # 軌道の長さチェック
    if len(actual_trace) < 3:
        issues.append("Too few trajectory points")
        score -= 30
    
    # 軌道の多様性チェック
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    
    if x_range < 50 or y_range < 50:
        issues.append("Limited spatial diversity")
        score -= 20
    
    # 軌道の滑らかさチェック
    distances = []
    for i in range(1, len(coords)):
        dist = np.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)
        distances.append(dist)
    
    if len(distances) > 1:
        distance_variance = np.var(distances)
        if distance_variance > 1000:  # 大きな分散は不規則な軌道を示す
            issues.append("Irregular trajectory spacing")
            score -= 15
    
    return {"quality_score": max(0, score), "issues": issues}

# 軌道品質評価
if trace and len(trace) > 0:
    quality_eval = evaluate_trajectory_quality(trace, img1)
    print(f"\n=== Trajectory Quality Evaluation ===")
    print(f"Quality Score: {quality_eval['quality_score']}/100")
    if quality_eval['issues']:
        print("Issues identified:")
        for issue in quality_eval['issues']:
            print(f"  - {issue}")
    else:
        print("No major issues identified")
else:
    print(f"\n=== Trajectory Quality Evaluation ===")
    print("Quality Score: 0/100")
    print("Issues identified:")
    print("  - No trajectory data available")

plt.show()
