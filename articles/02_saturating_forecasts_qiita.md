# Prophetチュートリアル第02回：成長限界編：飽和予測

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/notebooks/02_saturating_forecasts.ipynb)

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
