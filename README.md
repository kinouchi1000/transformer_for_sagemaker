# Sagemaker で Model Parallel ができるように Transofmer model を作る。

[ESPnet](https://github.com/espnet/espnet)を踏襲したディレクトリ構造となっている。

# requirement

requirements.txt にまとめてあるので、以下のコマンドでライブラリをダウンロードしてください。

```
pip install -r requirements.txt
```

# How to run the training


## データの準備

`src/test_utils/`以下に dump, stats, token_list, などを置く

## vscodeで学習の実行

`.vscode/launch.json`で必要なパラメータを記述しているので必要に応じて実行してください。
