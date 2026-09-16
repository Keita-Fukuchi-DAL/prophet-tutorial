# Prophetチュートリアル第04回：周期・イベント編：季節性と祝日効果

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keita-Fukuchi-DAL/prophet-tutorial/blob/main/04_seasonality_holiday_effects_and_regressors.ipynb)

## 祝日・イベント効果のモデリング（holidays）

祝日やスポーツの大会など、突発的にアクセスや売上がスパイクするイベント効果を含めるには、`holidays` データフレームを作成してモデルに渡します。データフレームには `holiday`（イベント名）、`ds`（日付）、および影響を受ける前後日数を示す `lower_window` と `upper_window` カラムを含めます。

```python
!pip install prophet
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from prophet import Prophet

playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16', '2010-01-24', '2010-02-07', '2011-01-08', '2012-01-08', '2012-01-14', '2013-01-12', '2014-01-12', '2014-01-19', '2014-02-02', '2015-01-11', '2016-01-17', '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
m = Prophet(holidays=holidays)
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
fig1.savefig('04_plot_holidays.png', bbox_inches='tight')

fig2 = m.plot_components(forecast)
fig2.savefig('04_plot_holidays_components.png', bbox_inches='tight')
```

![04_plot_holidays](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/04_plot_holidays.png)
![04_plot_holidays_components](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/04_plot_holidays_components.png)

## カスタム季節性の追加（add_seasonality）

Prophetは年・週・日のデフォルト季節性を持ちますが、月次周期（30.5日）など独自の周期を追加したい場合は `add_seasonality()` を使用します。`fourier_order` は季節性曲線の滑らかさ（高調波の数）を設定します。

```python
m_cust = Prophet(weekly_seasonality=False)
m_cust.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m_cust.fit(df)
forecast_cust = m_cust.predict(future)

fig3 = m_cust.plot_components(forecast_cust)
fig3.savefig('04_plot_custom_seasonality.png', bbox_inches='tight')
```

![04_plot_custom_seasonality](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/04_plot_custom_seasonality.png)

## 追加の回帰変数（add_regressor）

時系列自体のトレンドや季節性以外に、外部の数値因子（気温、マーケティング費用、特定条件フラグ等）を特徴量として予測モデルに組み込む場合は `add_regressor()` を使用します。未来データフレームにも該当回帰変数のデータが含まれている必要があります。

```python
def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    return 1 if (date.weekday() == 6 and date.month in [9, 10, 11, 12, 1]) else 0

df['nfl_sunday'] = df['ds'].apply(nfl_sunday)
m_reg = Prophet()
m_reg.add_regressor('nfl_sunday')
m_reg.fit(df)

future_reg = m_reg.make_future_dataframe(periods=365)
future_reg['nfl_sunday'] = future_reg['ds'].apply(nfl_sunday)
forecast_reg = m_reg.predict(future_reg)

fig4 = m_reg.plot_components(forecast_reg)
fig4.savefig('04_plot_regressor.png', bbox_inches='tight')
```

![04_plot_regressor](https://raw.githubusercontent.com/Keita-Fukuchi-DAL/prophet-tutorial/main/images/04_plot_regressor.png)

---
**連載ナビゲーション**
* [← 前の記事（第03回 トレンド編：変化点検知）](03_trend_changepoints_qiita.md)
* [📋 マスター記事（全10回目次）](master_article.md)
* [次の記事（第05回 変動増幅編：乗数的な季節性） →](05_multiplicative_seasonality_qiita.md)
