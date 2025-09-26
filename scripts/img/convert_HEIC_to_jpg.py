#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEIC画像をPNG画像に変換するプログラム
実行したディレクトリ内のすべてのHEIC画像をPNG形式に変換し、
最大長を700px以下にリサイズします。
"""

import os
import sys
from pathlib import Path
from PIL import Image
import pillow_heif
from tqdm import tqdm

def resize_image_if_needed(img, max_size=700):
    """
    画像の最大長が指定サイズ以下になるようリサイズする関数
    
    Args:
        img (PIL.Image): リサイズ対象の画像
        max_size (int): 最大長のピクセル数（デフォルト700）
        
    Returns:
        PIL.Image: リサイズされた画像
    """
    width, height = img.size
    max_dimension = max(width, height)
    
    if max_dimension <= max_size:
        return img
    
    # アスペクト比を保持してリサイズ
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img

def convert_heic_to_png(input_path, output_path, max_size=700):
    """
    HEIC画像をPNG画像に変換する関数（リサイズ機能付き）
    
    Args:
        input_path (str): 入力HEICファイルのパス
        output_path (str): 出力PNGファイルのパス
        max_size (int): 最大長のピクセル数（デフォルト700）
    """
    try:
        # HEICファイルを開く
        with Image.open(input_path) as img:
            original_size = img.size
            
            # PNGは透明度を保持できるので、RGBAモードを維持
            if img.mode == 'P':
                # パレットモードの場合はRGBAに変換
                img = img.convert('RGBA')
            elif img.mode not in ('RGBA', 'RGB', 'LA'):
                # その他のモードはRGBAに変換
                img = img.convert('RGBA')
            
            # 必要に応じてリサイズ
            img = resize_image_if_needed(img, max_size)
            new_size = img.size
            
            # PNGとして保存（透明度を保持）
            img.save(output_path, 'PNG', optimize=True)
            
            # リサイズ情報を表示
            if original_size != new_size:
                tqdm.write(f"変換完了: {input_path} -> {output_path} (リサイズ: {original_size[0]}x{original_size[1]} -> {new_size[0]}x{new_size[1]})")
            else:
                tqdm.write(f"変換完了: {input_path} -> {output_path}")
            return True
            
    except Exception as e:
        tqdm.write(f"エラー: {input_path} の変換に失敗しました - {str(e)}")
        return False

def find_heic_files(directory):
    """
    指定されたディレクトリ内のHEICファイルを検索する関数
    
    Args:
        directory (str): 検索するディレクトリのパス
        
    Returns:
        list: HEICファイルのパスのリスト
    """
    heic_extensions = ['.heic', '.HEIC', '.heif', '.HEIF']
    heic_files = []
    
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.suffix in heic_extensions:
            heic_files.append(file_path)
    
    return heic_files

def main():
    """メイン関数"""
    # pillow_heifを登録（HEICファイルを読み込むために必要）
    pillow_heif.register_heif_opener()
    
    # 現在のディレクトリを取得
    current_dir = "scripts/img/HEIC"
    print(f"変換対象ディレクトリ: {current_dir}")
    
    # HEICファイルを検索
    heic_files = find_heic_files(current_dir)
    
    if not heic_files:
        print("HEICファイルが見つかりませんでした。")
        return
    
    print(f"{len(heic_files)}個のHEICファイルが見つかりました。")
    
    # 変換処理
    success_count = 0
    for heic_file in tqdm(heic_files, desc="HEIC_PNG変換中", unit="files"):
        png_file = heic_file.with_suffix('.png')
        
        # 既にPNGファイルが存在する場合はスキップ
        if png_file.exists():
            tqdm.write(f"スキップ: {png_file} は既に存在します")
            continue
        
        # 変換実行（最大長700px以下にリサイズ）
        if convert_heic_to_png(str(heic_file), str(png_file).replace("HEIC/", ""), max_size=700):
            success_count += 1
    
    print(f"\n変換完了: {success_count}/{len(heic_files)} ファイルが正常に変換されました。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"予期しないエラーが発生しました: {str(e)}")
        sys.exit(1)
