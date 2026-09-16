# Prophetチュートリアル第10回：応用編：高度なトピック

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
