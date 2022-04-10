from re import X
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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

# パーセプトロンのオブジェクトの生成（インスタンス化)
ppn = Perceptron(eta=0.1, n_iter=10)
# 訓練データへのモデルの適合
ppn.fit(X, y)
# エポックと誤分類の関係を表す折れ線グラフのプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# 軸のラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Number of update')
# 図の表示
plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # 各特徴量を1次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとに訓練データをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')


# 決定領域のプロット
plot_decision_regions(X, y, classifier=ppn)
# 軸のラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定（左上に配置）
plt.legend(loc='upper left')
# 図の表示
plt.show()


class AdalineGD(object):
    # AdAptive LInear NEuron 分類器

    # 学習率(0.0より大きく1.0以下の値)
    eta: float
    # 訓練データの訓練回数
    n_iter: int
    # 重みを初期化するための乱数シード
    random_state: int

    # w_:1次元配列適用後の重み
    # cost_:各エポックでの誤差平方和のコスト関数

    def __init__(self, eta=0.01, n_iter=50, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # 訓練データに適合させる
        # X:{配列のようなデータ構造},shape={n_examples,n_features},訓練データ
        # n_examples:訓練データの個数、n_features:特徴量の個数
        # y:{配列のようなデータ構造}、shape={n_examples}
        # 戻り値:object

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            # 訓練回数分まで訓練データを反復
            net_input = self.net_input(X)
            output = self.activation(net_input)
            # 誤差の計算
            errors = (y-output)
            # w1.............wmの更新
            self.w_[1:] += self.eta*X.T.dot(errors)
            # w0の更新
            self.w_[0] += self.eta*errors.sum()
            # コスト関数の計算
            cost = (errors**2).sum()/2.0
            # コストの格納
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        # 総入力を計算
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        # 1ステップ後のクラスラベルを返す
        return np.where(self.activation(X) >= 0.0, 1, -1)


# 描画領域を1行2列に分割
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# 勾配降下法によるADALINEの学習(学習率 eta=0.01)
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
# エポック数とコストの関係を表す折れ線グラフのプロット(縦軸のコストは常用対数)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
# 軸のラベルの設定
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
# タイトルの設定
ax[0].set_title('Adaline-Learning rate 0.01')
# 勾配降下法によるADALINEの学習(学習率 eta=0.0001)
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
# エポック数とコストの関係を表す折れ線グラフのプロット
ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
# 軸のラベルの設定
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
# タイトルの設定
ax[1].set_title('Adaline-Learning rate 0.0001')
# 図の表示
plt.show()
