import os
import json
import glob
import pandas as pd
import numpy as np

# Prevent GUI popups
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_json, model_from_json
import warnings
warnings.filterwarnings('ignore')

os.makedirs('images', exist_ok=True)
os.makedirs('original_en', exist_ok=True)

print("Loading base datasets...")
manning_df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
retail_df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_retail_sales.csv')
yosemite_df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv')

def save_files(num_str, filename_base, nb_obj, md_str):
    nb_file = f"{num_str}_{filename_base}.ipynb"
    md_file = f"{num_str}_{filename_base}_qiita.md"
    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(nb_obj, f, ensure_ascii=False, indent=1)
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_str.strip() + "\n")
    print(f"Saved {nb_file} and {md_file}")

# ==========================================
# 02: saturating_forecasts
# ==========================================
print("Generating 02: saturating_forecasts...")
df_02 = manning_df.copy()
df_02['cap'] = 8.5
m_02 = Prophet(growth='logistic')
m_02.fit(df_02)
future_02 = m_02.make_future_dataframe(periods=1825)
future_02['cap'] = 8.5
fc_02 = m_02.predict(future_02)
fig = m_02.plot(fc_02)
fig.savefig('images/02_plot_saturating.png', bbox_inches='tight')
plt.close(fig)

df_02_floor = manning_df.copy()
df_02_floor['y'] = df_02_floor['y'] - 6
df_02_floor['cap'] = 6
df_02_floor['floor'] = 0
m_02_floor = Prophet(growth='logistic')
m_02_floor.fit(df_02_floor)
future_02_floor = m_02_floor.make_future_dataframe(periods=1825)
future_02_floor['cap'] = 6
future_02_floor['floor'] = 0
fc_02_floor = m_02_floor.predict(future_02_floor)
fig = m_02_floor.plot(fc_02_floor)
fig.savefig('images/02_plot_saturating_floor.png', bbox_inches='tight')
plt.close(fig)

nb_02 = {
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Prophetチュートリアル第02回：成長限界編：飽和予測\n\n## 成長モデルの定義（ロジスティック成長）\n\nProphetのデフォルトでは、トレンド予測に線形モデル（Linear Model）が使用されます。しかし、市場規模や利用可能な全人口など、理論上の上限（キャパシティ: carrying capacity）が存在する成長データを予測する場合、トレンドはその上限値に達すると飽和（Saturate）します。\n\nProphetでは、上限値（キャパシティ）を指定した[ロジスティック成長モデル](https://en.wikipedia.org/wiki/Logistic_function)を使用して予測を行うことが可能です。ここでは、Peyton ManningのWikipediaページビューの対数時系列データを用いて解説します。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!pip install prophet\nimport warnings\nwarnings.filterwarnings('ignore')\n\nimport pandas as pd\nfrom prophet import Prophet\n\ndf = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')\ndf['cap'] = 8.5"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## キャパシティ（cap）の設定と学習\n\nキャパシティ上限はデータフレームに `cap` カラムとして追加する必要があります。市場規模が拡大している場合は、`cap` を定数ではなく増加する時系列配列として設定することも可能です。\n\nモデルの定義時に `growth='logistic'` を指定し、過去データでモデルを学習させます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["m = Prophet(growth='logistic')\nm.fit(df)"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 未来予測における上限・下限（cap / floor）の設定\n\n未来の予測用データフレームを生成する際も、未来の各日付に対する `cap` を指定する必要があります。ここでは一定値 8.5 に固定し、今後5年間（1825日）を予測します。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["future = m.make_future_dataframe(periods=1825)\nfuture['cap'] = 8.5\nforecast = m.predict(future)\nfig1 = m.plot(forecast)\nfig1.savefig('02_plot_saturating.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 成長下限（Saturating Minimum / floor）を伴うモデル\n\n成長の下限（例えばゼロまたは特定の限界値）を指定する場合は `floor` 列を追加し、`growth='logistic'` と併用します。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["df['y'] = df['y'] - 6\ndf['cap'] = 6\ndf['floor'] = 0\nm_floor = Prophet(growth='logistic')\nm_floor.fit(df)\n\nfuture_floor = m_floor.make_future_dataframe(periods=1825)\nfuture_floor['cap'] = 6\nfuture_floor['floor'] = 0\nforecast_floor = m_floor.predict(future_floor)\nfig2 = m_floor.plot(forecast_floor)\nfig2.savefig('02_plot_saturating_floor.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 生成画像のZip一括ダウンロード\n\n以下のセルを実行すると、保存された画像がZip形式で一括ダウンロードされます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import glob, zipfile\nfrom google.colab import files\n\npng_files = glob.glob('*.png')\nwith zipfile.ZipFile('images.zip', 'w') as zipf:\n    for f in png_files:\n        zipf.write(f)\nfiles.download('images.zip')"]}
 ],
 "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2
}

md_02 = """# Prophetチュートリアル第02回：成長限界編：飽和予測

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/02_saturating_forecasts.ipynb)

## 成長モデルの定義（ロジスティック成長）

Prophetのデフォルトでは、トレンド予測に線形モデル（Linear Model）が使用されます。しかし、市場規模や利用可能な全人口など、理論上の上限（キャパシティ: carrying capacity）が存在する成長データを予測する場合、トレンドはその上限値に達すると飽和（Saturate）します。

Prophetでは、上限値（キャパシティ）を指定した[ロジスティック成長モデル](https://en.wikipedia.org/wiki/Logistic_function)を使用して予測を行うことが可能です。ここでは、Peyton ManningのWikipediaページビューの対数時系列データを用いて解説します。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df['cap'] = 8.5
```

## キャパシティ（cap）の設定と学習

キャパシティ上限はデータフレームに `cap` カラムとして追加する必要があります。市場規模が拡大している場合は、`cap` を定数ではなく増加する時系列配列として設定することも可能です。

モデルの定義時に `growth='logistic'` を指定し、過去データでモデルを学習させます。

```python
# ロジスティック成長モデルの学習
m = Prophet(growth='logistic')
m.fit(df)
```

## 未来予測における上限・下限（cap / floor）の設定

未来の予測用データフレームを生成する際も、未来の各日付に対する `cap` を指定する必要があります。ここでは一定値 8.5 に固定し、今後5年間（1825日）を予測します。

```python
future = m.make_future_dataframe(periods=1825)
future['cap'] = 8.5
forecast = m.predict(future)

fig1 = m.plot(forecast)
fig1.savefig('02_plot_saturating.png', bbox_inches='tight')
```

![02_plot_saturating](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/02_plot_saturating.png)

## 成長下限（Saturating Minimum / floor）を伴うモデル

成長の下限（例えばゼロまたは特定の限界値）を指定する場合は `floor` 列を追加し、`growth='logistic'` と併用します。

```python
df['y'] = df['y'] - 6
df['cap'] = 6
df['floor'] = 0
m_floor = Prophet(growth='logistic')
m_floor.fit(df)

future_floor = m_floor.make_future_dataframe(periods=1825)
future_floor['cap'] = 6
future_floor['floor'] = 0
forecast_floor = m_floor.predict(future_floor)

fig2 = m_floor.plot(forecast_floor)
fig2.savefig('02_plot_saturating_floor.png', bbox_inches='tight')
```

![02_plot_saturating_floor](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/02_plot_saturating_floor.png)

---
**連載ナビゲーション**
* [← 前の記事（第01回 基礎編：クイックスタート）](01_quick_start_qiita.md)
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第03回 トレンド編：変化点検知） →](03_trend_changepoints_qiita.md)
"""
save_files("02", "saturating_forecasts", nb_02, md_02)

# ==========================================
# 03: trend_changepoints
# ==========================================
print("Generating 03: trend_changepoints...")
df_03 = manning_df.copy()
m_03 = Prophet()
m_03.fit(df_03)
future_03 = m_03.make_future_dataframe(periods=365)
fc_03 = m_03.predict(future_03)
fig = m_03.plot(fc_03)
a = add_changepoints_to_plot(fig.gca(), m_03, fc_03)
fig.savefig('images/03_plot_changepoints.png', bbox_inches='tight')
plt.close(fig)

m_03_prior = Prophet(changepoint_prior_scale=0.5)
m_03_prior.fit(df_03)
fc_03_prior = m_03_prior.predict(future_03)
fig = m_03_prior.plot(fc_03_prior)
fig.savefig('images/03_plot_changepoints_prior.png', bbox_inches='tight')
plt.close(fig)

nb_03 = {
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Prophetチュートリアル第03回：トレンド編：変化点検知\n\n## トレンド転換点（変化点）の自動検知\n\nProphetは、時系列データ内のトレンドが変化する点（changepoints）を自動的に検知します。デフォルトでは、歴史データの最初の80%に対して変化点候補が自動配置され、モデル学習時にスパース性事前分布（L1正則化）を適用することで重要な変化点が選択されます。\n\n`add_changepoints_to_plot` 関数を使用すると、検知された変化点を赤い縦線としてグラフ上に描画できます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!pip install prophet\nimport warnings\nwarnings.filterwarnings('ignore')\n\nimport pandas as pd\nfrom prophet import Prophet\nfrom prophet.plot import add_changepoints_to_plot\n\ndf = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')\nm = Prophet()\nm.fit(df)\nfuture = m.make_future_dataframe(periods=365)\nforecast = m.predict(future)\n\nfig1 = m.plot(forecast)\na = add_changepoints_to_plot(fig1.gca(), m, forecast)\nfig1.savefig('03_plot_changepoints.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 変化点の柔軟性調整（changepoint_prior_scale）\n\nトレンドの変化に対してモデルがどの程度柔軟に追従するかは、ハイパーパラメータ `changepoint_prior_scale` で制御します。デフォルト値は `0.05` です。値を大きくするとトレンド追従がより柔軟（過学習リスク高）になり、小さくするとトレンド変化が過剰に抑制されます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# 変化点の柔軟性を高めたモデル (0.5)\nm_high = Prophet(changepoint_prior_scale=0.5)\nm_high.fit(df)\nforecast_high = m_high.predict(future)\n\nfig2 = m_high.plot(forecast_high)\nfig2.savefig('03_plot_changepoints_prior.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 生成画像のZip一括ダウンロード\n\n以下のセルを実行すると、保存された画像がZip形式で一括ダウンロードされます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import glob, zipfile\nfrom google.colab import files\n\npng_files = glob.glob('*.png')\nwith zipfile.ZipFile('images.zip', 'w') as zipf:\n    for f in png_files:\n        zipf.write(f)\nfiles.download('images.zip')"]}
 ],
 "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2
}

md_03 = """# Prophetチュートリアル第03回：トレンド編：変化点検知

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/03_trend_changepoints.ipynb)

## トレンド転換点（変化点）の自動検知

Prophetは、時系列データ内のトレンドが変化する点（changepoints）を自動的に検知します。デフォルトでは、歴史データの最初の80%に対して変化点候補が自動配置され、モデル学習時にスパース性事前分布（L1正則化）を適用することで重要な変化点が選択されます。

`add_changepoints_to_plot` 関数を使用すると、検知された変化点を赤い縦線としてグラフ上に描画できます。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast)
fig1.savefig('03_plot_changepoints.png', bbox_inches='tight')
```

![03_plot_changepoints](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/03_plot_changepoints.png)

## 変化点の柔軟性調整（changepoint_prior_scale）

トレンドの変化に対してモデルがどの程度柔軟に追従するかは、ハイパーパラメータ `changepoint_prior_scale` で制御します。デフォルト値は `0.05` です。値を大きくするとトレンド追従がより柔軟（過学習リスク高）になり、小さくするとトレンド変化が過剰に抑制されます。

```python
# 変化点の柔軟性を高めたモデル (0.5)
m_high = Prophet(changepoint_prior_scale=0.5)
m_high.fit(df)
forecast_high = m_high.predict(future)

fig2 = m_high.plot(forecast_high)
fig2.savefig('03_plot_changepoints_prior.png', bbox_inches='tight')
```

![03_plot_changepoints_prior](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/03_plot_changepoints_prior.png)

---
**連載ナビゲーション**
* [← 前の記事（第02回 成長限界編：飽和予測）](02_saturating_forecasts_qiita.md)
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第04回 周期・イベント編：季節性と祝日効果） →](04_seasonality_holiday_effects_and_regressors_qiita.md)
"""
save_files("03", "trend_changepoints", nb_03, md_03)

# ==========================================
# 04: seasonality_holiday_effects_and_regressors
# ==========================================
print("Generating 04: seasonality_holiday_effects_and_regressors...")
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16', '2010-01-24', '2010-02-07', '2011-01-08', '2012-01-08', '2012-01-14', '2013-01-12', '2014-01-12', '2014-01-19', '2014-02-02', '2015-01-11', '2016-01-17', '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))

df_04 = manning_df.copy()
m_04 = Prophet(holidays=holidays)
m_04.fit(df_04)
future_04 = m_04.make_future_dataframe(periods=365)
fc_04 = m_04.predict(future_04)

fig = m_04.plot(fc_04)
fig.savefig('images/04_plot_holidays.png', bbox_inches='tight')
plt.close(fig)

fig = m_04.plot_components(fc_04)
fig.savefig('images/04_plot_holidays_components.png', bbox_inches='tight')
plt.close(fig)

m_04_cust = Prophet(weekly_seasonality=False)
m_04_cust.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m_04_cust.fit(df_04)
fc_04_cust = m_04_cust.predict(future_04)
fig = m_04_cust.plot_components(fc_04_cust)
fig.savefig('images/04_plot_custom_seasonality.png', bbox_inches='tight')
plt.close(fig)

def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    return 1 if (date.weekday() == 6 and date.month in [9, 10, 11, 12, 1]) else 0

df_04['nfl_sunday'] = df_04['ds'].apply(nfl_sunday)
m_04_reg = Prophet()
m_04_reg.add_regressor('nfl_sunday')
m_04_reg.fit(df_04)

future_04_reg = m_04_reg.make_future_dataframe(periods=365)
future_04_reg['nfl_sunday'] = future_04_reg['ds'].apply(nfl_sunday)
fc_04_reg = m_04_reg.predict(future_04_reg)
fig = m_04_reg.plot_components(fc_04_reg)
fig.savefig('images/04_plot_regressor.png', bbox_inches='tight')
plt.close(fig)

nb_04 = {
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Prophetチュートリアル第04回：周期・イベント編：季節性と祝日効果\n\n## 祝日・イベント効果のモデリング（holidays）\n\n祝日やスポーツの大会など、突発的にアクセスや売上がスパイクするイベント効果を含めるには、`holidays` データフレームを作成してモデルに渡します。データフレームには `holiday`（イベント名）、`ds`（日付）、および影響を受ける前後日数を示す `lower_window` と `upper_window` カラムを含めます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!pip install prophet\nimport warnings\nwarnings.filterwarnings('ignore')\n\nimport pandas as pd\nfrom prophet import Prophet\n\nplayoffs = pd.DataFrame({\n  'holiday': 'playoff',\n  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16', '2010-01-24', '2010-02-07', '2011-01-08', '2012-01-08', '2012-01-14', '2013-01-12', '2014-01-12', '2014-01-19', '2014-02-02', '2015-01-11', '2016-01-17', '2016-01-24', '2016-02-07']),\n  'lower_window': 0,\n  'upper_window': 1,\n})\nsuperbowls = pd.DataFrame({\n  'holiday': 'superbowl',\n  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),\n  'lower_window': 0,\n  'upper_window': 1,\n})\nholidays = pd.concat((playoffs, superbowls))\n\ndf = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')\nm = Prophet(holidays=holidays)\nm.fit(df)\nfuture = m.make_future_dataframe(periods=365)\nforecast = m.predict(future)\n\nfig1 = m.plot(forecast)\nfig1.savefig('04_plot_holidays.png', bbox_inches='tight')\n\nfig2 = m.plot_components(forecast)\nfig2.savefig('04_plot_holidays_components.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## カスタム季節性の追加（add_seasonality）\n\nProphetは年・週・日のデフォルト季節性を持ちますが、月次周期（30.5日）など独自の周期を追加したい場合は `add_seasonality()` を使用します。`fourier_order` は季節性曲線の滑らかさ（高調波の数）を設定します。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["m_cust = Prophet(weekly_seasonality=False)\nm_cust.add_seasonality(name='monthly', period=30.5, fourier_order=5)\nm_cust.fit(df)\nforecast_cust = m_cust.predict(future)\n\nfig3 = m_cust.plot_components(forecast_cust)\nfig3.savefig('04_plot_custom_seasonality.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 追加の回帰変数（add_regressor）\n\n時系列自体のトレンドや季節性以外に、外部の数値因子（気温、マーケティング費用、特定条件フラグ等）を特徴量として予測モデルに組み込む場合は `add_regressor()` を使用します。未来データフレームにも該当回帰変数のデータが含まれている必要があります。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["def nfl_sunday(ds):\n    date = pd.to_datetime(ds)\n    return 1 if (date.weekday() == 6 and date.month in [9, 10, 11, 12, 1]) else 0\n\ndf['nfl_sunday'] = df['ds'].apply(nfl_sunday)\nm_reg = Prophet()\nm_reg.add_regressor('nfl_sunday')\nm_reg.fit(df)\n\nfuture_reg = m_reg.make_future_dataframe(periods=365)\nfuture_reg['nfl_sunday'] = future_reg['ds'].apply(nfl_sunday)\nforecast_reg = m_reg.predict(future_reg)\n\nfig4 = m_reg.plot_components(forecast_reg)\nfig4.savefig('04_plot_regressor.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 生成画像のZip一括ダウンロード\n\n以下のセルを実行すると、保存された画像がZip形式で一括ダウンロードされます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import glob, zipfile\nfrom google.colab import files\n\npng_files = glob.glob('*.png')\nwith zipfile.ZipFile('images.zip', 'w') as zipf:\n    for f in png_files:\n        zipf.write(f)\nfiles.download('images.zip')"]}
 ],
 "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2
}

md_04 = """# Prophetチュートリアル第04回：周期・イベント編：季節性と祝日効果

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/04_seasonality_holiday_effects_and_regressors.ipynb)

## 祝日・イベント効果のモデリング（holidays）

祝日やスポーツの大会など、突発的にアクセスや売上がスパイクするイベント効果を含めるには、`holidays` データフレームを作成してモデルに渡します。データフレームには `holiday`（イベント名）、`ds`（日付）、および影響を受ける前後日数を示す `lower_window` と `upper_window` カラムを含めます。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet

playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16', '2010-01-24', '2010-02-07', '2011-01-08', '2012-01-08', '2012-01-14', '2013-01-12', '2014-01-12', '2014-01-19', '2014-02-02', '2015-01-11', '2016-01-17', '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
m = Prophet(holidays=holidays)
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
fig1.savefig('04_plot_holidays.png', bbox_inches='tight')

fig2 = m.plot_components(forecast)
fig2.savefig('04_plot_holidays_components.png', bbox_inches='tight')
```

![04_plot_holidays](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/04_plot_holidays.png)
![04_plot_holidays_components](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/04_plot_holidays_components.png)

## カスタム季節性の追加（add_seasonality）

Prophetは年・週・日のデフォルト季節性を持ちますが、月次周期（30.5日）など独自の周期を追加したい場合は `add_seasonality()` を使用します。`fourier_order` は季節性曲線の滑らかさ（高調波の数）を設定します。

```python
m_cust = Prophet(weekly_seasonality=False)
m_cust.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m_cust.fit(df)
forecast_cust = m_cust.predict(future)

fig3 = m_cust.plot_components(forecast_cust)
fig3.savefig('04_plot_custom_seasonality.png', bbox_inches='tight')
```

![04_plot_custom_seasonality](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/04_plot_custom_seasonality.png)

## 追加の回帰変数（add_regressor）

時系列自体のトレンドや季節性以外に、外部の数値因子（気温、マーケティング費用、特定条件フラグ等）を特徴量として予測モデルに組み込む場合は `add_regressor()` を使用します。未来データフレームにも該当回帰変数のデータが含まれている必要があります。

```python
def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    return 1 if (date.weekday() == 6 and date.month in [9, 10, 11, 12, 1]) else 0

df['nfl_sunday'] = df['ds'].apply(nfl_sunday)
m_reg = Prophet()
m_reg.add_regressor('nfl_sunday')
m_reg.fit(df)

future_reg = m_reg.make_future_dataframe(periods=365)
future_reg['nfl_sunday'] = future_reg['ds'].apply(nfl_sunday)
forecast_reg = m_reg.predict(future_reg)

fig4 = m_reg.plot_components(forecast_reg)
fig4.savefig('04_plot_regressor.png', bbox_inches='tight')
```

![04_plot_regressor](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/04_plot_regressor.png)

---
**連載ナビゲーション**
* [← 前の記事（第03回 トレンド編：変化点検知）](03_trend_changepoints_qiita.md)
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第05回 変動増幅編：乗数的な季節性） →](05_multiplicative_seasonality_qiita.md)
"""
save_files("04", "seasonality_holiday_effects_and_regressors", nb_04, md_04)

# ==========================================
# 05: multiplicative_seasonality
# ==========================================
print("Generating 05: multiplicative_seasonality...")
df_05 = retail_df.copy()
m_05 = Prophet(seasonality_mode='multiplicative')
m_05.fit(df_05)
future_05 = m_05.make_future_dataframe(periods=365)
fc_05 = m_05.predict(future_05)

fig = m_05.plot(fc_05)
fig.savefig('images/05_plot_multiplicative.png', bbox_inches='tight')
plt.close(fig)

fig = m_05.plot_components(fc_05)
fig.savefig('images/05_plot_multiplicative_components.png', bbox_inches='tight')
plt.close(fig)

nb_05 = {
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Prophetチュートリアル第05回：変動増幅編：乗数的な季節性\n\n## 乗法的な季節性（Multiplicative Seasonality）\n\nProphetのデフォルトは加法モデル（Additive Model）であり、トレンドの上昇・下降に関わらず季節変動の振幅サイズは定数として扱われます。しかし実用上の時系列データ（売上や需要量）では、全体トレンドが成長するに伴って季節変動の幅も比例して大きくなる傾向が一般的です。\n\nこのようなデータには、`seasonality_mode='multiplicative'` を設定することで乗法モデルを適用し、トレンドに対する割合としての季節変動を正しくモデリングできます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!pip install prophet\nimport warnings\nwarnings.filterwarnings('ignore')\n\nimport pandas as pd\nfrom prophet import Prophet\n\ndf = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_retail_sales.csv')\nm = Prophet(seasonality_mode='multiplicative')\nm.fit(df)\nfuture = m.make_future_dataframe(periods=365)\nforecast = m.predict(future)\n\nfig1 = m.plot(forecast)\nfig1.savefig('05_plot_multiplicative.png', bbox_inches='tight')\n\nfig2 = m.plot_components(forecast)\nfig2.savefig('05_plot_multiplicative_components.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 生成画像のZip一括ダウンロード\n\n以下のセルを実行すると、保存された画像がZip形式で一括ダウンロードされます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import glob, zipfile\nfrom google.colab import files\n\npng_files = glob.glob('*.png')\nwith zipfile.ZipFile('images.zip', 'w') as zipf:\n    for f in png_files:\n        zipf.write(f)\nfiles.download('images.zip')"]}
 ],
 "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2
}

md_05 = """# Prophetチュートリアル第05回：変動増幅編：乗数的な季節性

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/05_multiplicative_seasonality.ipynb)

## 乗法的な季節性（Multiplicative Seasonality）

Prophetのデフォルトは加法モデル（Additive Model）であり、トレンドの上昇・下降に関わらず季節変動の振幅サイズは定数として扱われます。しかし実用上の時系列データ（売上や需要量）では、全体トレンドが成長するに伴って季節変動の幅も比例して大きくなる傾向が一般的です。

このようなデータには、`seasonality_mode='multiplicative'` を設定することで乗法モデルを適用し、トレンドに対する割合としての季節変動を正しくモデリングできます。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_retail_sales.csv')
m = Prophet(seasonality_mode='multiplicative')
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
fig1.savefig('05_plot_multiplicative.png', bbox_inches='tight')

fig2 = m.plot_components(forecast)
fig2.savefig('05_plot_multiplicative_components.png', bbox_inches='tight')
```

![05_plot_multiplicative](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/05_plot_multiplicative.png)
![05_plot_multiplicative_components](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/05_plot_multiplicative_components.png)

---
**連載ナビゲーション**
* [← 前の記事（第04回 周期・イベント編：季節性と祝日効果）](04_seasonality_holiday_effects_and_regressors_qiita.md)
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第06回 予測評価編：予測区間） →](06_uncertainty_intervals_qiita.md)
"""
save_files("05", "multiplicative_seasonality", nb_05, md_05)

# ==========================================
# 06: uncertainty_intervals
# ==========================================
print("Generating 06: uncertainty_intervals...")
df_06 = manning_df.copy()
m_06 = Prophet(interval_width=0.95)
m_06.fit(df_06)
future_06 = m_06.make_future_dataframe(periods=365)
fc_06 = m_06.predict(future_06)

fig = m_06.plot(fc_06)
fig.savefig('images/06_plot_uncertainty.png', bbox_inches='tight')
plt.close(fig)

nb_06 = {
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Prophetチュートリアル第06回：予測評価編：予測区間\n\n## 予測の不確実性区間（信頼区間・予測区間）の制御\n\nProphetは予測値 `yhat` の計算だけでなく、トレンドの将来の柔軟性変化による不確実性をシミュレーションして推定範囲（帯）を算出します。デフォルトでは80%の予測区間が指定されています。\n\n`interval_width` 引数（例: `0.95` で95%信頼区間）を変更することで、表示される不確実性の幅を制御できます。季節性の不確実性も加味したい場合は `mcmc_samples` パラメータでマルコフ連鎖モンテカルロ法（MCMC）サンプリングを実行します。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!pip install prophet\nimport warnings\nwarnings.filterwarnings('ignore')\n\nimport pandas as pd\nfrom prophet import Prophet\n\ndf = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')\nm = Prophet(interval_width=0.95)\nm.fit(df)\nfuture = m.make_future_dataframe(periods=365)\nforecast = m.predict(future)\n\nfig1 = m.plot(forecast)\nfig1.savefig('06_plot_uncertainty.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 生成画像のZip一括ダウンロード\n\n以下のセルを実行すると、保存された画像がZip形式で一括ダウンロードされます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import glob, zipfile\nfrom google.colab import files\n\npng_files = glob.glob('*.png')\nwith zipfile.ZipFile('images.zip', 'w') as zipf:\n    for f in png_files:\n        zipf.write(f)\nfiles.download('images.zip')"]}
 ],
 "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2
}

md_06 = """# Prophetチュートリアル第06回：予測評価編：予測区間

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/06_uncertainty_intervals.ipynb)

## 予測の不確実性区間（信頼区間・予測区間）の制御

Prophetは予測値 `yhat` の計算だけでなく、トレンドの将来の柔軟性変化による不確実性をシミュレーションして推定範囲（帯）を算出します。デフォルトでは80%の予測区間が指定されています。

`interval_width` 引数（例: `0.95` で95%信頼区間）を変更することで、表示される不確実性の幅を制御できます。季節性の不確実性も加味したい場合は `mcmc_samples` パラメータでマルコフ連鎖モンテカルロ法（MCMC）サンプリングを実行します。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
m = Prophet(interval_width=0.95)
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
fig1.savefig('06_plot_uncertainty.png', bbox_inches='tight')
```

![06_plot_uncertainty](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/06_plot_uncertainty.png)

---
**連載ナビゲーション**
* [← 前の記事（第05回 変動増幅編：乗数的な季節性）](05_multiplicative_seasonality_qiita.md)
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第07回 ノイズ対策編：外れ値の処理） →](07_outliers_qiita.md)
"""
save_files("06", "uncertainty_intervals", nb_06, md_06)

# ==========================================
# 07: outliers
# ==========================================
print("Generating 07: outliers...")
df_07 = manning_df.copy()
df_07.loc[(df_07['ds'] > '2010-01-01') & (df_07['ds'] < '2010-02-14'), 'y'] = None

m_07 = Prophet()
m_07.fit(df_07)
future_07 = m_07.make_future_dataframe(periods=365)
fc_07 = m_07.predict(future_07)

fig = m_07.plot(fc_07)
fig.savefig('images/07_plot_outliers.png', bbox_inches='tight')
plt.close(fig)

nb_07 = {
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Prophetチュートリアル第07回：ノイズ対策編：外れ値の処理\n\n## 突発的な外れ値（異常値）の影響除去\n\n実務データにおいて、システム障害やデータ集計ミス等による突発的な外れ値（Outliers）が含まれていると、トレンドや季節性の推定が大きく歪む原因になります。\n\nProphetは欠損値に対して強い堅牢性を持っているため、外れ値を無理に補間せず、対象期間の `y` の値を **`None` または `np.nan`（欠損値）** に設定して除去することが最善のソリューションとなります。Prophetは欠損区間を無視して正しくトレンドをモデリングします。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!pip install prophet\nimport warnings\nwarnings.filterwarnings('ignore')\n\nimport pandas as pd\nimport numpy as np\nfrom prophet import Prophet\n\ndf = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')\n# 異常値が含まれる特定の期間を None (欠損値) に指定して除去\ndf.loc[(df['ds'] > '2010-01-01') & (df['ds'] < '2010-02-14'), 'y'] = None\n\nm = Prophet()\nm.fit(df)\nfuture = m.make_future_dataframe(periods=365)\nforecast = m.predict(future)\n\nfig1 = m.plot(forecast)\nfig1.savefig('07_plot_outliers.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 生成画像のZip一括ダウンロード\n\n以下のセルを実行すると、保存された画像がZip形式で一括ダウンロードされます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import glob, zipfile\nfrom google.colab import files\n\npng_files = glob.glob('*.png')\nwith zipfile.ZipFile('images.zip', 'w') as zipf:\n    for f in png_files:\n        zipf.write(f)\nfiles.download('images.zip')"]}
 ],
 "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2
}

md_07 = """# Prophetチュートリアル第07回：ノイズ対策編：外れ値の処理

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/07_outliers.ipynb)

## 突発的な外れ値（異常値）の影響除去

実務データにおいて、システム障害やデータ集計ミス等による突発的な外れ値（Outliers）が含まれていると、トレンドや季節性の推定が大きく歪む原因になります。

Prophetは欠損値に対して強い堅牢性を持っているため、外れ値を無理に補間せず、対象期間の `y` の値を **`None` または `np.nan`（欠損値）** に設定して除去することが最善のソリューションとなります。Prophetは欠損区間を無視して正しくトレンドをモデリングします。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from prophet import Prophet

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

# 異常値が含まれる特定の期間を None (欠損値) に指定して除去
df.loc[(df['ds'] > '2010-01-01') & (df['ds'] < '2010-02-14'), 'y'] = None

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
fig1.savefig('07_plot_outliers.png', bbox_inches='tight')
```

![07_plot_outliers](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/07_plot_outliers.png)

---
**連載ナビゲーション**
* [← 前の記事（第06回 予測評価編：予測区間）](06_uncertainty_intervals_qiita.md)
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第08回 データ形式編：非日次データ） →](08_non-daily_data_qiita.md)
"""
save_files("07", "outliers", nb_07, md_07)

# ==========================================
# 08: non-daily_data
# ==========================================
print("Generating 08: non-daily_data...")
df_08_m = retail_df.copy()
m_08_m = Prophet()
m_08_m.fit(df_08_m)
future_08_m = m_08_m.make_future_dataframe(periods=24, freq='MS')
fc_08_m = m_08_m.predict(future_08_m)
fig = m_08_m.plot(fc_08_m)
fig.savefig('images/08_plot_monthly.png', bbox_inches='tight')
plt.close(fig)

df_08_sub = yosemite_df.copy()
m_08_sub = Prophet(changepoint_prior_scale=0.01)
m_08_sub.fit(df_08_sub)
future_08_sub = m_08_sub.make_future_dataframe(periods=300, freq='h')
fc_08_sub = m_08_sub.predict(future_08_sub)
fig = m_08_sub.plot_components(fc_08_sub)
fig.savefig('images/08_plot_subdaily.png', bbox_inches='tight')
plt.close(fig)

nb_08 = {
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Prophetチュートリアル第08回：データ形式編：非日次データ\n\n## 月次データ・非日次データのモデリング\n\nProphetは日次（Daily）データだけでなく、月次（Monthly）データやサブデイリー（時間・分単位）データもサポートしています。\n\n月次データの場合は `make_future_dataframe` の `freq` 引数に `'MS'`（Month Start）等のPandasオフセットを指定して未来データフレームを構築します。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!pip install prophet\nimport warnings\nwarnings.filterwarnings('ignore')\n\nimport pandas as pd\nfrom prophet import Prophet\n\ndf_monthly = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_retail_sales.csv')\nm = Prophet()\nm.fit(df_monthly)\nfuture_monthly = m.make_future_dataframe(periods=24, freq='MS')\nforecast_monthly = m.predict(future_monthly)\n\nfig1 = m.plot(forecast_monthly)\nfig1.savefig('08_plot_monthly.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## サブデイリー（時間単位・分単位）データと日中周期性\n\nタイムスタンプ形式 `YYYY-MM-DD HH:MM:SS` を含んだ高頻度データでは、一日の時間帯ごとの変化を表す **`daily_seasonality`（日中季節性）** が自動的に有効化され可視化されます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["df_subdaily = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv')\nm_sub = Prophet(changepoint_prior_scale=0.01)\nm_sub.fit(df_subdaily)\nfuture_sub = m_sub.make_future_dataframe(periods=300, freq='h')\nforecast_sub = m_sub.predict(future_sub)\n\nfig2 = m_sub.plot_components(forecast_sub)\nfig2.savefig('08_plot_subdaily.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 生成画像のZip一括ダウンロード\n\n以下のセルを実行すると、保存された画像がZip形式で一括ダウンロードされます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import glob, zipfile\nfrom google.colab import files\n\npng_files = glob.glob('*.png')\nwith zipfile.ZipFile('images.zip', 'w') as zipf:\n    for f in png_files:\n        zipf.write(f)\nfiles.download('images.zip')"]}
 ],
 "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2
}

md_08 = """# Prophetチュートリアル第08回：データ形式編：非日次データ

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/08_non-daily_data.ipynb)

## 月次データ・非日次データのモデリング

Prophetは日次（Daily）データだけでなく、月次（Monthly）データやサブデイリー（時間・分単位）データもサポートしています。

月次データの場合は `make_future_dataframe` の `freq` 引数に `'MS'`（Month Start）等のPandasオフセットを指定して未来データフレームを構築します。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet

df_monthly = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_retail_sales.csv')
m = Prophet()
m.fit(df_monthly)
future_monthly = m.make_future_dataframe(periods=24, freq='MS')
forecast_monthly = m.predict(future_monthly)

fig1 = m.plot(forecast_monthly)
fig1.savefig('08_plot_monthly.png', bbox_inches='tight')
```

![08_plot_monthly](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/08_plot_monthly.png)

## サブデイリー（時間単位・分単位）データと日中周期性

タイムスタンプ形式 `YYYY-MM-DD HH:MM:SS` を含んだ高頻度データでは、一日の時間帯ごとの変化を表す **`daily_seasonality`（日中季節性）** が自動的に有効化され可視化されます。

```python
df_subdaily = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv')
m_sub = Prophet(changepoint_prior_scale=0.01)
m_sub.fit(df_subdaily)
future_sub = m_sub.make_future_dataframe(periods=300, freq='h')
forecast_sub = m_sub.predict(future_sub)

fig2 = m_sub.plot_components(forecast_sub)
fig2.savefig('08_plot_subdaily.png', bbox_inches='tight')
```

![08_plot_subdaily](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/08_plot_subdaily.png)

---
**連載ナビゲーション**
* [← 前の記事（第07回 ノイズ対策編：外れ値の処理）](07_outliers_qiita.md)
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第09回 精度検証編：モデル診断） →](09_diagnostics_qiita.md)
"""
save_files("08", "non-daily_data", nb_08, md_08)

# ==========================================
# 09: diagnostics
# ==========================================
print("Generating 09: diagnostics...")
df_09 = manning_df.copy()
m_09 = Prophet()
m_09.fit(df_09)

# Perform quick CV for image generation
df_cv = cross_validation(m_09, initial='730 days', period='180 days', horizon='365 days')
df_p = performance_metrics(df_cv)

fig = plot_cross_validation_metric(df_cv, metric='rmse')
fig.savefig('images/09_plot_cross_validation.png', bbox_inches='tight')
plt.close(fig)

fig2 = plot_cross_validation_metric(df_cv, metric='mape')
fig2.savefig('images/09_plot_metrics.png', bbox_inches='tight')
plt.close(fig2)

nb_09 = {
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Prophetチュートリアル第09回：精度検証編：モデル診断\n\n## クロスバリデーション（交差検証）の自動実行\n\nProphetは、過去データ上でモデル精度を客観評価するためのクロスバリデーション（交差検証: Simulated Historical Forecasts）を自動化する `cross_validation` 関数を提供します。\n\n`initial`（初期学習期間）、`period`（評価ウィンドウの間隔）、`horizon`（予測スパン）を指定して交差検証を実行します。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!pip install prophet\nimport warnings\nwarnings.filterwarnings('ignore')\n\nimport pandas as pd\nfrom prophet import Prophet\nfrom prophet.diagnostics import cross_validation, performance_metrics\nfrom prophet.plot import plot_cross_validation_metric\n\ndf = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')\nm = Prophet()\nm.fit(df)\n\n# クロスバリデーションの実行\ndf_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')\ndf_cv.head()"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 評価指標（Performance Metrics）と可視化\n\n`performance_metrics` 関数を使用すると、RMSE、MAE、MAPEなどの標準的な評価指標を算出できます。また `plot_cross_validation_metric` で予測地平（Horizon）に対する誤差の推移を散布図と青い移動平均線で視覚化できます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["df_p = performance_metrics(df_cv)\nprint(df_p.head())\n\nfig1 = plot_cross_validation_metric(df_cv, metric='rmse')\nfig1.savefig('09_plot_cross_validation.png', bbox_inches='tight')\n\nfig2 = plot_cross_validation_metric(df_cv, metric='mape')\nfig2.savefig('09_plot_metrics.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 生成画像のZip一括ダウンロード\n\n以下のセルを実行すると、保存された画像がZip形式で一括ダウンロードされます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import glob, zipfile\nfrom google.colab import files\n\npng_files = glob.glob('*.png')\nwith zipfile.ZipFile('images.zip', 'w') as zipf:\n    for f in png_files:\n        zipf.write(f)\nfiles.download('images.zip')"]}
 ],
 "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2
}

md_09 = """# Prophetチュートリアル第09回：精度検証編：モデル診断

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/09_diagnostics.ipynb)

## クロスバリデーション（交差検証）の自動実行

Prophetは、過去データ上でモデル精度を客観評価するためのクロスバリデーション（交差検証: Simulated Historical Forecasts）を自動化する `cross_validation` 関数を提供します。

`initial`（初期学習期間）、`period`（評価ウィンドウの間隔）、`horizon`（予測スパン）を指定して交差検証を実行します。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
m = Prophet()
m.fit(df)

# クロスバリデーションの実行
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')
df_cv.head()
```

## 評価指標（Performance Metrics）と可視化

`performance_metrics` 関数を使用すると、RMSE、MAE、MAPEなどの標準的な評価指標を算出できます。また `plot_cross_validation_metric` で予測地平（Horizon）に対する誤差の推移を散布図と青い移動平均線で視覚化できます。

```python
df_p = performance_metrics(df_cv)
print(df_p.head())

fig1 = plot_cross_validation_metric(df_cv, metric='rmse')
fig1.savefig('09_plot_cross_validation.png', bbox_inches='tight')

fig2 = plot_cross_validation_metric(df_cv, metric='mape')
fig2.savefig('09_plot_metrics.png', bbox_inches='tight')
```

![09_plot_cross_validation](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/09_plot_cross_validation.png)
![09_plot_metrics](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/09_plot_metrics.png)

---
**連載ナビゲーション**
* [← 前の記事（第08回 データ形式編：非日次データ）](08_non-daily_data_qiita.md)
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第10回 応用編：高度なトピック） →](10_additional_topics_qiita.md)
"""
save_files("09", "diagnostics", nb_09, md_09)

# ==========================================
# 10: additional_topics
# ==========================================
print("Generating 10: additional_topics...")
df_10 = manning_df.copy()
m_10 = Prophet(growth='flat')
m_10.fit(df_10)
future_10 = m_10.make_future_dataframe(periods=365)
fc_10 = m_10.predict(future_10)

fig = m_10.plot(fc_10)
fig.savefig('images/10_plot_flat_trend.png', bbox_inches='tight')
plt.close(fig)

nb_10 = {
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Prophetチュートリアル第10回：応用編：高度なトピック\n\n## フラットトレンド（growth='flat'）の設定\n\n時系列に成長や衰退の長期的な傾きが存在せず、特定の水準（平均値）の周りを一定に変動している場合は `growth='flat'` を指定して定常的なトレンドモデルを適用します。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!pip install prophet\nimport warnings\nwarnings.filterwarnings('ignore')\n\nimport pandas as pd\nfrom prophet import Prophet\nfrom prophet.serialize import model_to_json, model_from_json\n\ndf = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')\nm = Prophet(growth='flat')\nm.fit(df)\nfuture = m.make_future_dataframe(periods=365)\nforecast = m.predict(future)\n\nfig1 = m.plot(forecast)\nfig1.savefig('10_plot_flat_trend.png', bbox_inches='tight')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## モデルの保存・再利用（Serialization）\n\n実務運用において、学習済みProphetモデルを保存・共有・再読み込みするには `prophet.serialize` モジュール（JSONフォーマット）を使用します。Pythonの標準 `pickle` ではなく JSON 形式を使用することで、異なるバージョン環境間でも安全にシリアライズが可能です。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# モデルのJSON保存\nwith open('serialized_model.json', 'w') as fout:\n    fout.write(model_to_json(m))\n\n# JSONからの復元\nwith open('serialized_model.json', 'r') as fin:\n    m_restored = model_from_json(fin.read())\n\nprint('Model successfully restored from JSON!')"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 生成画像のZip一括ダウンロード\n\n以下のセルを実行すると、保存された画像がZip形式で一括ダウンロードされます。"]},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import glob, zipfile\nfrom google.colab import files\n\npng_files = glob.glob('*.png')\nwith zipfile.ZipFile('images.zip', 'w') as zipf:\n    for f in png_files:\n        zipf.write(f)\nfiles.download('images.zip')"]}
 ],
 "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2
}

md_10 = """# Prophetチュートリアル第10回：応用編：高度なトピック

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/10_additional_topics.ipynb)

## フラットトレンド（growth='flat'）の設定

時系列に成長や衰退の長期的な傾きが存在せず、特定の水準（平均値）の周りを一定に変動している場合は `growth='flat'` を指定して定常的なトレンドモデルを適用します。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
m = Prophet(growth='flat')
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
fig1.savefig('10_plot_flat_trend.png', bbox_inches='tight')
```

![10_plot_flat_trend](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/10_plot_flat_trend.png)

## モデルの保存・再利用（Serialization）

実務運用において、学習済みProphetモデルを保存・共有・再読み込みするには `prophet.serialize` モジュール（JSONフォーマット）を使用します。Pythonの標準 `pickle` ではなく JSON 形式を使用することで、異なるバージョン環境間でも安全にシリアライズが可能です。

```python
# モデルのJSON保存
with open('serialized_model.json', 'w') as fout:
    fout.write(model_to_json(m))

# JSONからの復元
with open('serialized_model.json', 'r') as fin:
    m_restored = model_from_json(fin.read())

print('Model successfully restored from JSON!')
```

---
**連載ナビゲーション**
* [← 前の記事（第09回 精度検証編：モデル診断）](09_diagnostics_qiita.md)
* [📋 マスター記事（全10回目次）](master_article.md)
"""
save_files("10", "additional_topics", nb_10, md_10)

print("ALL TUTORIAL FILES AND IMAGES SUCCESSFULLY GENERATED!")
