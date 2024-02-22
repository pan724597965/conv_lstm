import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Layer, Bidirectional, LSTM
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization


# 创建滑动窗口序列数据
def create_sliding_window_data(data, feature_columns, target_column, sequence_length, step_size=1):
    features = data[feature_columns].values
    target = data[target_column].values.reshape(-1, 1)

    # 数据标准化
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target)

    X, y = [], []
    for i in range(0, len(features_scaled) - sequence_length, step_size):
        X.append(features_scaled[i:i + sequence_length, :])
        y.append(target_scaled[i + sequence_length, 0])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


# 自定义 RMSE 指标函数
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


def predict_with_model(data, feature_columns, target_column, sequence_length, epochs=100, only_predict=True):
    # 读取数据集
    if only_predict:
        data = pd.read_csv('./data/total6_10.csv', parse_dates=['valuechangetime'])
    else:
        data = pd.read_csv('./data/total6_30day.csv', parse_dates=['valuechangetime'])
    data.set_index('valuechangetime', inplace=True)

    # 创建序列数据
    X, y, scaler = create_sliding_window_data(data, feature_columns, target_column, sequence_length)
    file_name = f"{sequence_length}_{epochs}_{target_column}"

    # 创建 Conv + LSTM 模型
    model_filename = file_name + '_model.h5'

    if os.path.exists(model_filename):
        print("=====>> 模型加载成功：", model_filename)
        # 如果文件存在，加载模型
        with tf.keras.utils.custom_object_scope({'TransformerAttention': TransformerAttention}):
            model = tf.keras.models.load_model(model_filename, custom_objects={'rmse': rmse})
    else:
        model = tf.keras.models.Sequential([
                tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                                    strides=1, padding="causal",
                                    activation="relu",
                                    input_shape=(sequence_length, len(feature_columns))),
                Bidirectional(LSTM(32, return_sequences=True)),
                Bidirectional(LSTM(32, return_sequences=True)),
                TransformerAttention(d_model=32, num_heads=8),
                Bidirectional(LSTM(32, return_sequences=False)),
                tf.keras.layers.Dense(1)
        ])
        # 编译模型
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', rmse])

    if not only_predict:
        # 划分训练集和测试集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 创建 EarlyStopping 回调
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)

        # 训练模型
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        model.save(model_filename)

    # 进行预测
    prediction_scaled = model.predict(X[-sequence_length:])
    prediction = scaler.inverse_transform(prediction_scaled)

    return prediction

class MultiFeatureAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(MultiFeatureAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(name="W_q", shape=(input_shape[-1], input_shape[-1]),
                                  initializer="uniform", trainable=True)
        self.W_k = self.add_weight(name="W_k", shape=(input_shape[-1], input_shape[-1]),
                                  initializer="uniform", trainable=True)
        self.W_v = self.add_weight(name="W_v", shape=(input_shape[-1], input_shape[-1]),
                                  initializer="uniform", trainable=True)

        super(MultiFeatureAttentionLayer, self).build(input_shape)

    def call(self, x):
        q = tf.matmul(x, self.W_q)
        k = tf.matmul(x, self.W_k)
        v = tf.matmul(x, self.W_v)

        attn_scores = tf.matmul(q, k, transpose_b=True)
        attn_scores = tf.nn.softmax(attn_scores, axis=-1)

        output = tf.matmul(attn_scores, v)
        return output

class TransformerAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(TransformerAttention, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.attention(inputs, inputs)
        return self.norm(x + inputs)

