# Prophetチュートリアル第05回：変動増幅編：乗数的な季節性

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/notebooks/05_multiplicative_seasonality.ipynb)

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
* [マスター記事（全10回目次）](master_article.md)
* [次の記事（第06回 予測評価編：予測区間） →](06_uncertainty_intervals_qiita.md)
