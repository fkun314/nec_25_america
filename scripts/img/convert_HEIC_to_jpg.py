#!/usr/bin/env python3
"""
HEIC画像をPNG画像に変換するプログラム
実行したディレクトリ内のすべてのHEIC画像をPNG形式に変換します。
"""

import os
import sys
from pathlib import Path
from PIL import Image
import pillow_heif
from tqdm import tqdm

def convert_heic_to_png(input_path, output_path):
    """
    HEIC画像をPNG画像に変換する関数
    
    Args:
        input_path (str): 入力HEICファイルのパス
        output_path (str): 出力PNGファイルのパス
    """
    try:
        # HEICファイルを開く
        with Image.open(input_path) as img:
            # PNGは透明度を保持できるので、RGBAモードを維持
            if img.mode == 'P':
                # パレットモードの場合はRGBAに変換
                img = img.convert('RGBA')
            elif img.mode not in ('RGBA', 'RGB', 'LA'):
                # その他のモードはRGBAに変換
                img = img.convert('RGBA')
            
            # PNGとして保存（透明度を保持）
            img.save(output_path, 'PNG', optimize=True)
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
    for heic_file in tqdm(heic_files, desc="HEIC→PNG変換中", unit="ファイル"):
        # 出力ファイル名を生成（拡張子を.pngに変更）
        png_file = heic_file.with_suffix('.png')
        
        # 既にPNGファイルが存在する場合はスキップ
        if png_file.exists():
            tqdm.write(f"スキップ: {png_file} は既に存在します")
            continue
        
        # 変換実行
        if convert_heic_to_png(str(heic_file), str(png_file).replace("HEIC/", "")):
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
