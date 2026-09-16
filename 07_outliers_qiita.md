# Prophetチュートリアル第07回：ノイズ対策編：外れ値の処理

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
