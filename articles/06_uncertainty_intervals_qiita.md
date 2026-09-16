# Prophetチュートリアル第06回：予測評価編：予測区間

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/notebooks/06_uncertainty_intervals.ipynb)

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
