from cProfile import label
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron(object):
    # パーセプトロンの分類器

    # 学習率(0.0より大きく1.0以下の値)
    eta: float

    # 訓練データの回数
    n_iter: int

    # 重みを初期化するための乱数シード
    random_state: int

    def __init__(self, eta=0.01, n_iter=50, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # 訓練データに適合させる
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # 重みw1......wmの更新
                updata = self.eta*(target-self.predict(xi))
                self.w_[1:] += updata*xi
                # 重みw0の更新
                self.w_[0] += updata
                # 重みの更新が0でない場合ご分類としてカウント
                errors += int(updata != 0.0)
            # 反復ごとの誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        # 総入力を計算
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # 1ステップ後のクラスラベルを返す
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# irisデータセットのURL
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

print('URL:', s)
# irisデータセットのcsvファイルを読み込む
df = pd.read_csv(s, header=None, encoding='utf-8')
# 最後の5行を出力
print(df.tail())

# 1-100行目の目的変数の抽出
y = df.iloc[0:100, 4].values

# Iris-setosaを-1,Iris-versicolorを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)
# 1-100行目の1,3列目を抽出
X = df.iloc[0:100, [0, 2]].values
# 品種setosaのプロット(赤の〇)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# 品種versicolorのプロット(青の×)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
            marker='x', label='versicolor')
# 軸ラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定
plt.legend(loc='upper left')
# 図の表示
plt.show()
