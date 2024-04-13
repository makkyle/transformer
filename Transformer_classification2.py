# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:52:37 2024

@author: 10249
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras import activations
# from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.models import Model



X1=np.load('./X1_antigen_antibody_bind__antigen_5688_300_20.npy')
X2=np.load('./X2_antigen_antibody_bind__light_5688_300_20.npy')
X3=np.load('./X3_antigen_antibody_bind__heavy_5688_300_20.npy')
Y=np.load('./Y_antigen_antibody_bind__bind_label_y_5688_1.npy')


x_train=np.hstack([X1,X2,X3])
print('x shape',x_train.shape)
y_train=Y



n_classes = len(np.unique(y_train))
# print(x_train.shape,x_test.shape)
print(n_classes)
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,):
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)





input_shape = x_train.shape[1:]

print('input_shape',input_shape)
model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
# model.summary()
#tf.keras.utils.plot_model(model, to_file='Transformer_classification.png', show_shapes=True)

checkpoint_filepath = './best/transformer_c2_{loss:.5f}_{val_loss:.5f}.weights.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

# callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
callbacks = [tf_callback,model_checkpoint_callback]
model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=200,
    callbacks=callbacks,
)
model.save('./transformer_class2.h5')
# model.evaluate(x_test, y_test, verbose=1)


