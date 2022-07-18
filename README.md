# 画像からの汚れ検出

画像からの衣服汚れ検出用プログラムの説明書

作成：2022/05/09 Keisuke

## Requirements

開発した当時のバージョンは以下である

- matplotlib==3.5.1
- numpy==1.20.3
- python==3.7.11
- pytorch==1.7.1
- torchvision==0.8.2
- tqdm==4.62.3
- scikit-image==0.18.1
- scikit-learn==1.0.2

## 使い方

以下、全ての記述は ./BRAE をカレントディレクトリとしている前提で進める

- カレントディレクトリが異なる場合はcdコマンドでカレントディレクトリに移動してください


1. 以下のコマンドを実行をすると学習、推論(異常検知)を実行できる

```:bash
$ python main.py
```

2. モデルの学習時、パラメータを変更したい場合はコマンドライン引数で記述する。コマンドライン引数で指定されないパラメータはデフォルトのものが使用される。指定可能なパラメータは次章で説明する。

```:bash
$ python main.py --unet_dim 32 --ae_dim 32
```

3. --train_skipを使うと学習済みのモデルを使った検証のみを実行できる

```:bash
$ python main.py --train_skip
```


## 指定可能なパラメータ(main.py)

### 学習設定


| parameter | 説明 | default |
| - | - | - |
| seed | 初期シード値の変更 | 111 |
| train_skip | 学習をスキップして動く。 | false |
| result_path | 結果が出力されるディレクトリの名前を指定する。 | results |
| model_path | モデルの重みが出力されるディレクトリの名前を指定する。 | models |
| epochs | 学習epoch数を指定する | 200 |

### モデル設定


| parameter | 説明 | default |
| - | - | - |
| unet_model | U-Net部で使用するモデルを指定する。指定可能なモデルは modules/model.py で定義される。 | unet |
| unet_dim | モデルのU-Net部のボトルネックチャンネル数を指定する。他の層のチャンネル数は、入力の畳み込み後のチャンネル数を基準として解像度が上がるごとに倍になる。 | 8 |
| ae_model | Autoencoder部で使用するモデルを指定する。指定可能なモデルは modules/model.py で定義される。 | aeI |
| ae_dim | モデルのAutoencoder部のボトルネックチャンネル数を指定する。他の層のチャンネル数は、入力の畳み込み後のチャンネル数を基準として解像度が上がるごとに倍になる。 | 16 |
| init_type | モデルの初期化方法を指定する。指定可能な初期化方法は ./modules/initializer.py で定義される。 | ne_uni |
| lr | 学習率を指定する。 | 0.001 |
| scheduler | スケジューラーを指定する。 | LR |

### データセット設定


| parameter | 説明 | default |
| - | - | - |
| dataset | 使用するデータセットを指定する。指定可能なデータセットは ./config/dataset/ 以下に設定ファイルが存在し、 ./src/datasets/ 以下に対応する torch.utils.data.Datasetクラスのコードと pytorch_lightning.LightningDataModuleクラスのコードが存在し、 ./dataset/ 以下に実データが格納されている必要がある。 | cloth |
| data_type | データセットのカテゴリを指定する。 | pants |
| resize | データセットのリサイズの解像度を指定する。 | 256 |
| batch_size | バッチサイズを指定する。 | 32 |

### 異常検知用設定


| parameter | 説明 | default |
| - | - | - |
| ssim_rate | 異常検知時に用いる異常度算におけるSSIMの割合を指定する。 | 0 |

## ファイルについて

### ファイル構成

* BRAE
  * data
    * pants
      * ground_truth
      * test
      * train
  * modules
    * initializer.py
    * loader.py
    * logger.py
    * losses.py
    * model.py
    * utils.py
  * ae_train.py
  * concat_train.py
  * main.py
  * separate_train.py
  * softmax_train.py
  * test.py

### ファイルの内容


| file/directory | 説明 |
| - | - |
| data/ | データセット置き場 |
| models/ | main.pyを実行すると自動作成されるモデルの重み保存フォルダ |
| modules/ | フォルダ |
| results/ | main.pyを実行すると自動作成される結果フォルダ |
| ae_train.py | Autoencoderを学習するコード |
| concat_train.py | 提案モデルBを学習するコード |
| main.py | すべてを動かす元となるコード |
| separate_train.py | パイプラインモデルを学習するコード |
| softmax_train.py | 提案モデルAを学習するコード |
| test.py | 異常検知をするコード |
| modules/initializer.py | 重みの初期化方法 |
| modules/loader.py | データセット定義 |
| modules/logger.py | loggerの初期設定 |
| modules/losses.py | 損失関数 |
| modules/model.py | モデル定義 |
| modules/utils.py | 結果画像の保存、AUROCやIoUなどのスコア計算、学習曲線の出力・描画クラスなど色々 |

## 出力について

```:bash
[INFO] {'ae_dim': 16, 'ae_model': 'aeI', 'batch_size': 32, 'data_type': 'pants', 'dataset': 'cloth', 'epochs': 200, 'init_type': 'he_uni', 'lr': 0.001, 'model': 'concat', 'model_path': 'models/cloth256LR_concat/unet8aeI16/seed111_batch32he_uni', 'resize': 256, 'result_dir': 'results/cloth256LR_concat/unet8_aeI16/seed111_batch32he_uni', 'result_path': 'results', 'scheduler': 'LR', 'seed': 111, 'split_len': 5, 'split_num': 2, 'ssim_rate': 0.5, 'start_epoch': 1, 'train': False, 'unet': True, 'unet_dim': 8, 'unet_model': 'unet'}
# パラメータを表示している

[INFO] device : cuda
# 使用デバイスを表す。cudaが望ましい

[INFO] fold:2 epoch:200
# データの拾い方とEpoch数

Test: 100%|_______________| 4/4 [00:04<00:00,  1.01s/batch] 
# テストデータの実行

[INFO] pixel PRAUC: 0.0371
# PRAUC

[INFO] pixel AUROC: 0.9801
# AUROC

[INFO] TPR:0.9415873015873016 ,FPR:0.073764062832775, threshold:0.2036505788564682
# TPRとFPRのバランスが最良のときのTPRとFPR

[INFO] TPR:1.0 ,FPR:0.30016406782765176, threshold:0.04977850243449211
# TPRが1.0のときの最良のFPR

Plt: 100%|________________| 110/110 [00:16<00:00,  6.74plt/s](
# 画像出力
```