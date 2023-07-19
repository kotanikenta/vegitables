import codecs
import io

from flask import Flask, render_template, request
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model

app = Flask(__name__)

def preprocess_image(image_path):
    with codecs.open(image_path, 'rb') as file:
        data = file.read()
    image = Image.open(io.BytesIO(data))
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image_path):
    model = load_model('vegetable_model.keras')  # モデルの読み込み
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    results = [(class_name, probability) for _, class_name, probability in decoded_predictions]
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image.save('static/uploaded_image.jpg')
    results = predict_image('static/uploaded_image.jpg')
    return render_template('results.html', results=results)

# データセットのパス
dataset_path = 'static/images/'

# クラスのリスト
classes = ['broccori', 'cabbage', 'carotto', 'cucumber', 'eggplant', 'Okura', 'onion', 'potate', 'tomate']

# データセットの分割とデータ拡張の設定
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 訓練データの生成器
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# テストデータの生成器
test_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# VGG16モデルの読み込み
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# トップ層の追加
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

# 新しいモデルの作成
model = Model(inputs=base_model.input, outputs=predictions)

# モデルのコンパイル
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# モデルの訓練（初回のみ実行）
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.n // test_generator.batch_size
)

# モデルの保存（初回のみ実行）
model.save('vegetable_model.keras')

# Flaskアプリをポート5000で実行
app.run(debug=True, port=5000)
