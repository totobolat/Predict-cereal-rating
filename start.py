import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd


df = pd.read_csv('./input/cereal.csv')
print(df.head())

rating = df['rating']
calories = df['calories']
protein = df['protein']
fat = df['fat']
sodium = df['sodium']
fiber = df['fiber']
carbo = df['carbo']
sugars = df['sugars']
potass = df['potass']
vitamins = df['vitamins']
ingredients = protein + fat + (sodium/100) + fiber + carbo + sugars + (potass/100)

def plot_features() : 
    fig, axs = plt.subplots(ncols=2,nrows=2,figsize=(14, 7), layout='constrained')
    axs[0,0].set_xlabel('ingredients')
    axs[0,0].set_ylabel('calories')
    axs[0,0].set_title('Ingredients to Calories')
    axs[0,0].scatter(ingredients,calories, c='red',marker='x')
    axs[1,0].scatter(calories,rating, c='blue',marker='x')
    axs[1,0].set_xlabel('calories')
    axs[1,0].set_ylabel('rating')
    axs[1,0].set_title('Calories to Rating')
    axs[0,1].scatter(sugars,rating, c='green',marker='x')
    axs[0,1].set_xlabel('sugar')
    axs[0,1].set_ylabel('rating')
    axs[0,1].set_title('Sugar to Rating')
    axs[1,1].scatter(ingredients,rating, c='yellow',marker='x')
    axs[1,1].set_xlabel('ingredients')
    axs[1,1].set_ylabel('rating')
    axs[1,1].set_title('Ingredients to Rating')
    plt.show()

#plot_features()

df = df.drop(['protein','fat','sodium','fiber','carbo','potass','vitamins','name','mfr','type','shelf','weight','cups'],axis=1)
df['ingredients'] = ingredients


train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('rating')
test_labels = test_features.pop('rating')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.asarray(train_features).astype('float32'))
print(normalizer.mean.numpy())


model = Sequential(
    [
        normalizer,
        Dense(32, activation = 'relu'),
        Dense(16, activation = 'relu'),
        Dense(1, activation = 'linear')
    ]
)

model.compile(
    loss='mean_absolute_error',
    optimizer=tf.keras.optimizers.Adam(0.01),
)

history = model.fit(
    train_features,train_labels,
    epochs=200,
    verbose = 0,
    validation_split = 0.2
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print('hist tail  : ',hist.tail())

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error Rating')
  plt.legend()
  plt.grid(True)
  plt.show()
plot_loss(history)

test_results = {}
test_results['model'] = model.evaluate(
    test_features, test_labels, verbose=0)
print('test resuls',test_results['model'])
pred_test_data = pd.DataFrame({
    "calories": [60,50],
    "sugar": [1,0],
    "ingredients": [20,30]
})
print('predictions : ')
print(model.predict(pred_test_data))
print('predictions : ')
print(model.predict(train_features.iloc[:3]))
