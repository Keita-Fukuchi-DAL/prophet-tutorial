# Prophetチュートリアル第03回：トレンド編：変化点検知

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/notebooks/03_trend_changepoints.ipynb)

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
* [マスター記事（全10回目次）](master_article.md)
* [次の記事（第04回 周期・イベント編：季節性と祝日効果） →](04_seasonality_holiday_effects_and_regressors_qiita.md)
