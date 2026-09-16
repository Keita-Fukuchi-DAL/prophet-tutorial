# Prophetチュートリアル第09回：精度検証編：モデル診断

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/notebooks/09_diagnostics.ipynb)

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
