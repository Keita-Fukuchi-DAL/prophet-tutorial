# Prophetチュートリアル第01回：基礎編：クイックスタート

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/01_quick_start.ipynb)

## Python API

Prophetは`sklearn`のモデルAPIに準拠しています。`Prophet`クラスのインスタンスを作成し、その`fit`メソッドおよび`predict`メソッドを呼び出します。

Prophetへの入力は、常に`ds`と`y`の2つのカラムを持つデータフレームである必要があります。`ds`（日付スタンプ）カラムは、Pandasが期待するフォーマット（日付の場合は`YYYY-MM-DD`、タイムスタンプの場合は`YYYY-MM-DD HH:MM:SS`）である必要があります。`y`カラムは数値であり、予測したい測定値を表します。

例として、[Peyton Manning](https://en.wikipedia.org/wiki/Peyton_Manning)のWikipediaページにおける日別ページビュー数の対数時系列データを見てみましょう。このデータはRの[Wikipediatrend](https://cran.r-project.org/package=wikipediatrend)パッケージを使用してスクレイピングされたものです。Peyton Manningのデータは、複数の季節性、変化する成長率、特別な日（プレーオフやスーパーボウルへの出場など）をモデリングする能力など、Prophetの機能のいくつかを示しているため、優れた例となります。CSVファイルは[こちら](https://github.com/facebook/prophet/blob/main/examples/example_wp_log_peyton_manning.csv)で入手できます。

まず、データのインポートと環境構築を行います。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet
```

```python
df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df.head()
```

新しい`Prophet`オブジェクトをインスタンス化してモデルを適合（fit）させます。予測手順の設定はコンストラクタに渡されます。次に、`fit`メソッドを呼び出し、過去のデータフレームを渡します。適合には1〜5秒かかります。

```python
m = Prophet()
m.fit(df)
```

予測は、予測対象の日付を含む`ds`カラムを持つデータフレームに対して行われます。ヘルパーメソッド`Prophet.make_future_dataframe`を使用すると、指定した日数だけ未来に拡張された適切なデータフレームを取得できます。デフォルトでは過去の日付も含まれるため、モデルの適合状況も確認できます。

```python
future = m.make_future_dataframe(periods=365)
future.tail()
```

`predict`メソッドは、`future`の各行に対して`yhat`という名前の予測値を割り当てます。過去の日付を渡すと、サンプル内適合が提供されます。ここでの`forecast`オブジェクトは、予測値を含む`yhat`カラムや、要素成分および不確実性区間のカラムを含む新しいデータフレームです。

```python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

`Prophet.plot`メソッドを呼び出し、予測データフレームを渡すことで、予測をプロットできます。

```python
fig1 = m.plot(forecast)
fig1.savefig('01_plot_forecast.png', bbox_inches='tight')
```

![01_plot_forecast](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/01_plot_forecast.png)

予測要素を確認したい場合は、`Prophet.plot_components`メソッドを使用できます。デフォルトでは、時系列のトレンド、年間の季節性、および週ごとの季節性が表示されます。祝日を含めた場合は、それらもここに表示されます。

```python
fig2 = m.plot_components(forecast)
fig2.savefig('01_plot_components.png', bbox_inches='tight')
```

![01_plot_components](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/01_plot_components.png)

---
**連載ナビゲーション**
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第02回 成長限界編：飽和予測） →](#)
