import pandas as pd

# CSVファイルのパス
file_path = 'band_gap_results.csv'

# CSVファイルを読み込む
data = pd.read_csv(file_path)

# 欠損値を含む行を削除する
cleaned_data = data.dropna()
cleaned_data = cleaned_data[cleaned_data['Band Gap (eV)'] >= 0.005]
# クリーンなデータを新しいCSVファイルとして保存する
cleaned_file_path = 'band_gap_results_cleaned.csv'
cleaned_data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")