# perceptron
パーセプトロンの実装をしてみました。Irisデータセットに含まれている品種の訓練データの分布状況と、2つの特徴量軸(花びらの長さ、がく辺の長さ)を
表している。この2次元の特徴量部分空間では、Iris-setosaとIris-versicolorを分類するのに線形の決定境界で十分であることがわかる。

このグラフからわかるように、6回目のエポックの後、パーセプトロンはすでに収束しており、訓練データを完璧に分類できるようになる。
パーセプトロンのアルゴリズムが決定境界を学習したことがわかる。

Iris訓練サブセットに含まれている品種の訓練データをすべて分類できる。ADALINEの学習アルゴリズムについても実装しました。
