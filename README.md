
## 概要

手書きの画像を入力し、漢数字の一～十を識別します。
深層学習プログラミングの学習を目的として作成しました。

ラベルデータの用意からモデルの作成、学習、推論までの一通りの流れを確認できます。

### 実行環境

Jupyter-Notebook上で動作します。

### リポジトリ

ラベルデータを含めて、GitHubの以下のリポジトリとして公開しています。

* https://github.com/lhideki/kansuji-classifier

### ディレクトリ構成

* label・・・教師データです。ディレクトリ名をラベル名として認識します。ラベル名に対応するディレクトリ配下に対応する画像データを配置しています。
* target・・・推論フェーズで使用するテストデータです。targetディレクトリに配下の画像を読み取り、それぞれどの分類になるかを推論します。

### 説明の流れ

以降はソースコード、その直後に解説という構成でプログラムを説明しています。
全体の構成は以下のとおりです。

* ライブラリのインポート
* 関数定義
    * 画像を読み込んでベクトルに変換するための関数
    * ディレクトリを指定してラベルデータを読み込むための関数
    * モデルの作成と学習をするための関数
* 学習フェーズ
* テストデータによる評価
* 推論フェーズ

## ライブラリのインポート

以下のライブラリをインポートしています。

* PIL・・・画像処理ライブラリ。画像のリサイズや、グレイスケールへの変換で使用。
* numpy・・・言わずと知れた行列計算用ライブラリ。
* os・・・ファイルやディレクトリのパスの操作で使用。
* glob・・・ファイルやディレクトリを検索するためのユーティリティ。ワイルドカードで一括検索などが可能。便利。
* keras・・・TensorFlowなどのディープラーニング用エンジンを簡単に利用するためのフレームワーク。凄い。
* sklearn・・・scikit learn。機械学習で使用する色々な機能を提供。ここでは、交差検証のために開発データとテストデータの分離、および評価結果を表示するための関数を利用。


```python
from PIL import Image
import numpy as Np
import os as Os
import glob as Glob
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.datasets import mnist as Mnist
from keras import utils as Utils
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
```

    Using TensorFlow backend.


## 画像を読み込んでベクトルに変換するための関数

引数に指定された画像ファイルのパスを読み込み、以下の処理を行います。

* リサイズ(グローバル変数のimage_width、 image_heightで指定したサイズにリサイズ)
* グレイスケールに変換
* 2値化(白: 0、黒: 1)

最後にベクトル(1次元配列)に変形したndarrayを返します。


```python
def convert_image_to_vector(image_filepath):
    global image_width, image_height
    
    image = Image.open(image_filepath, 'r')
    resized_image = image.resize((image_width, image_height))
    gray_image = resized_image.convert('L')
    onehot_image = gray_image.point(lambda x: 1 if x < 150 else 0)

    array = Np.reshape(onehot_image, (image_width * image_height))

    return array
```

## ディレクトリを指定してラベルデータを読み込むための関数

ラベルデータを保存したディレクトリのパスを受け取り、以下の3つの値を返します。

* vectors・・・画像をベクトル化したデータ。
* labels・・・ラベルの配列。vectorsの配列要素の順序に対応したラベルが保存されている。
* printable_labels・・・ラベルデータが保存されているディレクトリ名を保存したもの。推論した後に返されるのは、ラベルの配列の要素番号であるため、それを人がわかる形に変換するために使用する。


```python
def load_labeldata(label_path):
    label_dirs = Glob.glob(Os.path.join(label_path, '*'))
    labels = []
    vectors = []
    printable_labels = {}
    class_count= len(label_dirs)

    for i, label_dir in enumerate(label_dirs):
        printable_labels[i] = Os.path.basename(label_dir)
        image_dirs = Glob.glob(Os.path.join(label_dir, '*.jpg'))
        for t, image_file in enumerate(image_dirs):
            vector = convert_image_to_vector(image_file)
            labels.append(i)
            vectors.append(vector)
    
    reshaped_labels = Utils.np_utils.to_categorical(labels, num_classes = class_count)
    reshaped_vectors = Np.asarray(vectors)

    return {
        'vectors': reshaped_vectors,
        'labels': reshaped_labels,
        'printable_labels': printable_labels
    }
```

## モデルの作成と学習をするための関数

class_countとして受け取った分類をするためのモデルを作成し、ラベルデータから学習を行います。

作成するモデルは、全結合とDropoutを交互に重ねた3層構造です。


```python
def create_model(class_count, train_labels, train_vectors):
    global image_width
    global image_height
    
    model = Sequential()
    model.add(Dense(units = 512, activation = 'relu', input_dim = (image_width * image_height)))
    model.add(Dropout(0.6))
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = class_count, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',
                optimizer = 'rmsprop',
                 metrics = ['acc'])
    tensor_board = TensorBoard(log_dir = 'tflog')
    model.fit(train_vectors, train_labels, verbose = 1, epochs = 30, callbacks = [tensor_board])
    model.summary()
    
    return model
```

## 学習フェーズ

学習に渡す前に開発データとテストデータ8:2の比率で別け、それぞれをモデルに渡して学習を行います。


```python
from sklearn.metrics import classification_report, confusion_matrix

image_width = 100
image_height= int(image_width * 0.87)

label_data = load_labeldata('label')
tmp_data = train_test_split(label_data['vectors'], label_data['labels'], train_size = 0.8, test_size = 0.2)
train_vectors, test_vectors, train_labels, test_labels = map(lambda vec: Np.asarray(vec), tmp_data)
class_count = len(label_data['printable_labels'])
 
model = create_model(class_count, train_labels, train_vectors)
```

    WARNING:tensorflow:From /Users/hideki/.pyenv/versions/3.6.4/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py:198: retry (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use the retry module or similar alternatives.
    Epoch 1/30
    155/155 [==============================] - 0s 3ms/step - loss: 1.4653 - acc: 0.5161
    Epoch 2/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.1174 - acc: 0.9677
    Epoch 3/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0253 - acc: 1.0000
    Epoch 4/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0081 - acc: 1.0000
    Epoch 5/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0069 - acc: 1.0000
    Epoch 6/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0056 - acc: 1.0000
    Epoch 7/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0027 - acc: 1.0000
    Epoch 8/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0135 - acc: 0.9935
    Epoch 9/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0021 - acc: 1.0000
    Epoch 10/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0010 - acc: 1.0000
    Epoch 11/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0034 - acc: 1.0000
    Epoch 12/30
    155/155 [==============================] - 0s 2ms/step - loss: 4.4356e-04 - acc: 1.0000
    Epoch 13/30
    155/155 [==============================] - 0s 2ms/step - loss: 6.0559e-04 - acc: 1.0000
    Epoch 14/30
    155/155 [==============================] - 0s 2ms/step - loss: 9.4518e-04 - acc: 1.0000
    Epoch 15/30
    155/155 [==============================] - 0s 2ms/step - loss: 4.2408e-04 - acc: 1.0000
    Epoch 16/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0010 - acc: 1.0000
    Epoch 17/30
    155/155 [==============================] - 0s 2ms/step - loss: 8.3154e-05 - acc: 1.0000
    Epoch 18/30
    155/155 [==============================] - 0s 2ms/step - loss: 5.9973e-04 - acc: 1.0000
    Epoch 19/30
    155/155 [==============================] - 0s 2ms/step - loss: 5.9789e-04 - acc: 1.0000
    Epoch 20/30
    155/155 [==============================] - 0s 2ms/step - loss: 5.7693e-05 - acc: 1.0000
    Epoch 21/30
    155/155 [==============================] - 0s 2ms/step - loss: 9.5239e-05 - acc: 1.0000
    Epoch 22/30
    155/155 [==============================] - 0s 2ms/step - loss: 2.5447e-04 - acc: 1.0000
    Epoch 23/30
    155/155 [==============================] - 0s 2ms/step - loss: 2.2102e-04 - acc: 1.0000
    Epoch 24/30
    155/155 [==============================] - 0s 3ms/step - loss: 4.2461e-05 - acc: 1.0000
    Epoch 25/30
    155/155 [==============================] - 0s 3ms/step - loss: 1.2166e-04 - acc: 1.0000
    Epoch 26/30
    155/155 [==============================] - 0s 2ms/step - loss: 0.0128 - acc: 0.9935
    Epoch 27/30
    155/155 [==============================] - 0s 3ms/step - loss: 8.1067e-06 - acc: 1.0000
    Epoch 28/30
    155/155 [==============================] - 0s 2ms/step - loss: 3.4963e-05 - acc: 1.0000
    Epoch 29/30
    155/155 [==============================] - 0s 3ms/step - loss: 9.1907e-07 - acc: 1.0000
    Epoch 30/30
    155/155 [==============================] - 0s 2ms/step - loss: 3.0052e-06 - acc: 1.0000
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 512)               4454912   
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               262656    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 4,722,698
    Trainable params: 4,722,698
    Non-trainable params: 0
    _________________________________________________________________


## テストデータによる評価

テストデータに対して推論を行い、精度、検出率、F1値を算出しています。


```python
import pandas as pd

#score = model.evaluate(test_vectors, test_labels)
#print(score)
#print('Loss = ', score[0])
#print('Accuracy = ', score[1])

pred_labels = model.predict_classes(test_vectors)
numeric_labels = [i for i in test_labels.argmax(axis = 1)]
print(classification_report(numeric_labels, pred_labels, target_names = label_data['printable_labels'].values()))
```

                 precision    recall  f1-score   support
    
              十       1.00      1.00      1.00         4
              二       1.00      1.00      1.00         2
              三       1.00      1.00      1.00         6
              一       1.00      1.00      1.00         4
              四       1.00      1.00      1.00         5
              八       1.00      1.00      1.00         5
              五       1.00      1.00      1.00         3
              九       1.00      1.00      1.00         1
              六       1.00      1.00      1.00         5
              七       1.00      1.00      1.00         4
    
    avg / total       1.00      1.00      1.00        39
    


## 推論フェーズ

`target`ディレクトリにある画像をファイルに対して推論を行い、結果を出力しています。


```python
def get_label(result, labels):
    return labels[result.argmax()]

test_dirs = Glob.glob(Os.path.join('target', '*.jpg'))

report = []
files = []
for i, image in enumerate(test_dirs):
    vector = convert_image_to_vector(image)
    result = model.predict(Np.array([vector]))
    label = get_label(result[0], label_data['printable_labels'])
    files.append(image)
    report.append({
        'label': label,
        'probability': result.max()
    })

r = pd.DataFrame(report, index = files)
r
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>target/IMG_0206.jpg</th>
      <td>五</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>target/IMG_0261.jpg</th>
      <td>六</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>target/IMG_0262.jpg</th>
      <td>七</td>
      <td>0.999998</td>
    </tr>
    <tr>
      <th>target/IMG_0263.jpg</th>
      <td>八</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>target/IMG_0264.jpg</th>
      <td>九</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>target/IMG_0265.jpg</th>
      <td>十</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>target/paint_4.jpg</th>
      <td>四</td>
      <td>0.196821</td>
    </tr>
    <tr>
      <th>target/paint_5.jpg</th>
      <td>三</td>
      <td>0.284940</td>
    </tr>
    <tr>
      <th>target/IMG_0195.jpg</th>
      <td>四</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>target/IMG_0194.jpg</th>
      <td>三</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>target/IMG_0193.jpg</th>
      <td>二</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>target/IMG_0192.jpg</th>
      <td>一</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


