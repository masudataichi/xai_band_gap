import pandas as pd
from mp_api.client import MPRester

# APIキーを入力してください
api_key = 'Rawr0UsQwACsr3DeuqGeh7kO28t9qzFL'

compounds =pd.read_csv("band_gap_download.csv")["composition"].to_list()

results = []


try:
    # MPResterを使用してデータにアクセス
    with MPRester(api_key) as mpr:

        for compound in compounds:
            # get_entriesメソッドを使用してデータを取得
            docs = mpr.summary.search(formula=[compound], fields=["material_id", "band_gap", "formula_pretty", "formation_energy_per_atom"])

            if docs:
                # 生成エンタルピーが最も小さいエントリを選択
                min_energy_doc = min(docs, key=lambda x: x.formation_energy_per_atom)
                compound = min_energy_doc.formula_pretty
                material_id = min_energy_doc.material_id
                band_gap = min_energy_doc.band_gap
                formation_energy = min_energy_doc.formation_energy_per_atom
                results.append({
                    "Formula": compound,
                    "Material ID": material_id,
                    "Band Gap (eV)": band_gap,
                    "Formation Energy (eV/atom)": formation_energy,
                })
            else:
                results.append({
                    "Formula": compound,
                    "Material ID": "N/A",
                    "Band Gap (eV)": "N/A",
                    "Formation Energy (eV/atom)": "N/A",
                })
except Exception as e:
    print(f"エラーが発生しました: {e}")

# データフレームを作成
df = pd.DataFrame(results)
print(df)

# CSVファイルに出力
csv_file = "band_gap_results.csv"
df.to_csv(csv_file, index=False)

print(f"結果が{csv_file}に保存されました。")