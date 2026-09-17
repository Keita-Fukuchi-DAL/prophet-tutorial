# Prophetチュートリアル第08回：データ形式編：非日次データ

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/notebooks/08_non-daily_data.ipynb)

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
* [マスター記事（全10回目次）](master_article.md)
* [次の記事（第09回 精度検証編：モデル診断） →](09_diagnostics_qiita.md)
