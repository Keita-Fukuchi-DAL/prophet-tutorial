# Prophetチュートリアル第01回：基礎編：クイックスタート

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/01_quick_start.ipynb)

## Python API

Meta（旧Facebook）が開発した時系列予測ライブラリ **Prophet** の操作感は、Pythonの機械学習ライブラリ `scikit-learn` のモデルAPIと統一されています。

基本フローは非常にシンプルで、`Prophet` クラスのインスタンスを生成した後、過去データを `fit()` メソッドで学習させ、`predict()` メソッドで未来の予測値を算出します。

## 入力データの形式（dsとyの制約）

Prophetに投入する入力データフレームには、**`ds`** と **`y`** という名前の2つのカラムが必須となります。

- **`ds`（datestamp）**: 日付または日時の識別カラム。Pandasが認識できる日付フォーマット（日付のみの場合は `YYYY-MM-DD`、時間を含む場合は `YYYY-MM-DD HH:MM:SS`）に揃える必要があります。
- **`y`**: 予測対象となる数値データ（売上高、アクセス数、需要量など）。

### サンプルデータセットの解説
本チュートリアルではサンプルとして、NFL（米アメフトリーグ）の名クォーターバックである **Peyton Manning（ペイトン・マニング）** のWikipediaページにおける日別ページビュー（PV）数の対数時系列データ（`example_wp_log_peyton_manning.csv`）を使用します。

このデータはRの `wikipediatrend` パッケージで取得されたもので、アクセス数の急激なスパイクによる影響を和らげスケールを揃えるために対数変換（`log`）されています。Peyton Manningのアクセスデータは、以下のような時系列分析における重要な要素がすべて含まれているため、Prophetの機能を学ぶのに最も適したサンプルデータです。

- **複数の季節性**: 試合が開催される日曜日ごとの周期（週季節性）や、シーズン中の年季節性
- **トレンドの転換点**: 現役生活の経過や引退に伴う長期的な成長率の変化
- **イベント効果**: プレーオフやスーパーボウル出場日など、特定のイベント日における急激なアクセス増

まず、必要なライブラリのインポートと前処理を行います。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet
```

```python
# サンプルデータの取得
df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df.head()
```

## モデルのインスタンス化と学習 (`fit`)

モデルの構築は、`Prophet` オブジェクトをインスタンス化することから始まります。予測手順に関するハイパーパラメータ（成長モデルや季節性の詳細設定など）は、このコンストラクタに渡します。

モデルオブジェクトを生成後、`fit()` メソッドに過去のデータフレーム `df` を渡すことで、モデルの学習（フィッティング）が実行されます。通常、学習処理は1〜5秒程度で完了します。

```python
# モデルの定義とフィッティング
m = Prophet()
m.fit(df)
```

## 未来の予測用データフレームの生成 (`make_future_dataframe`)

モデルの学習が完了したら、予測を行いたい対象の期間（日付スタンプ）を含むデータフレームを用意します。

Prophetが提供するヘルパー関数 `make_future_dataframe()` を使用すると、指定した日数（`periods`）だけ未来に拡張された日付列を持つデータフレームを容易に生成できます。デフォルトでは過去の学習期間の日付も自動的に含まれるため、過去データへのフィッティング状況（インサンプル評価）もあわせて確認できます。

```python
# 今後365日分（1年間）の未来の日付を含むデータフレームを生成
future = m.make_future_dataframe(periods=365)
future.tail()
```

## 予測の実行と推論結果 (`predict`)

作成した `future` データフレームを `predict()` メソッドに渡すことで、各日付に対する予測処理（推論）を実行します。

算出される予測結果オブジェクト `forecast` は、主となる予測値 **`yhat`** をはじめ、予測の不確実性の幅を示す予測区間（下限 **`yhat_lower`**、上限 **`yhat_upper`**）、およびトレンドや季節性などの分解成分のカラムを含む新しいデータフレームです。

```python
# 予測（推論）の実行
forecast = m.predict(future)

# 予測結果の主要カラムの確認
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

## 予測結果のプロットと可視化 (`plot`)

`m.plot()` メソッドに予測データフレーム `forecast` を渡すことで、過去の実測値（黒点）、予測モデルの推定トレンドおよび不確実性区間（青い線と青い帯）を直感的に可視化できます。

```python
# 予測結果のグラフ描画と画像保存
fig1 = m.plot(forecast)
fig1.savefig('01_plot_forecast.png', bbox_inches='tight')
```

![01_plot_forecast](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/01_plot_forecast.png)

## 時系列の変動成分の分解可視化 (`plot_components`)

Prophetの強力な機能の一つに、予測値を **「全体トレンド」「曜日ごとの季節性」「年間の季節性」** などの要素に分解して可視化できる点があります。

`m.plot_components()` メソッドを呼び出すことで、それぞれの成分が全体の予測値にどのように影響を与えているかを詳細に分析できます。もしモデルに祝日やイベント効果を追加した場合は、それらの影響度も独立した成分としてプロットされます。

```python
# 分解成分（トレンド・季節性内訳）のグラフ描画と画像保存
fig2 = m.plot_components(forecast)
fig2.savefig('01_plot_components.png', bbox_inches='tight')
```

![01_plot_components](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/01_plot_components.png)

---
**連載ナビゲーション**
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第02回 成長限界編：飽和予測） →](#)
