import tensorflow as tf
from keras.engine import data_adapter

class TemporalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.E1 = units
    def build(self, input_shape):
        # h (bs, T, U)
        self.Wa = self.add_weight("Wa", shape=(self.E1, input_shape[-1]), initializer='random_normal', trainable=True)
        self.ba = self.add_weight("ba", shape=(self.E1, 1), initializer='zeros', trainable=True)
        self.ua = self.add_weight("ua", shape=(self.E1, 1), initializer='random_normal', trainable=True)
        return super().build(input_shape)

    def call(self, h):
        # h (bs, T, U) -> (bs, T, U, 1)
        x = tf.expand_dims(h, axis = -1)
        x = tf.matmul(self.Wa, x) + self.ba 
        x = tf.nn.tanh(x)
        x = tf.matmul(self.ua, x, transpose_a= True)[..., 0, 0]
        alpha = tf.nn.softmax(x) #(bs, T)
        a_s = tf.matmul(tf.expand_dims(alpha, axis=-1), h, transpose_a = True)[:,0,:]
        return a_s # (bs, U)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class LatentLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def build(self, input_shape):
        self.concat = tf.keras.layers.Concatenate(axis= -1)
        self.linear_layer = tf.keras.layers.Dense(1, activation = "linear")
        return super().build(input_shape)

    def call(self, inputs):
        alpha, h = inputs
        e = tf.keras.layers.concatenate([alpha, h[:, -1, :]], axis= -1)
        return e

class PredictionLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def build(self, input_shape):
        self.concat = tf.keras.layers.Concatenate(axis= -1)
        self.linear_layer = tf.keras.layers.Dense(1, activation = "linear")
        return super().build(input_shape)

    def call(self, e):
        y = self.linear_layer(e)[..., 0]
        return y

    
class AdvALSTMModel(tf.keras.Model):
    def __init__(self, my_beta, my_epsilon, my_L2, my_E, my_U, my_E1):
        super().__init__()

        self.my_beta = my_beta
        self.my_epsilon = my_epsilon 
        self.my_E = my_E
        self.my_U = my_U
        self.my_E1 = my_E1
        self.my_L2 = my_L2
        
        self.feature_mapping_layer = tf.keras.layers.Dense(units = self.my_E, activation= "tanh")
        self.lstm_layer = tf.keras.layers.LSTM(units= self.my_U, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(self.my_L2))
        self.temporal_attention_layer = TemporalAttentionLayer(units= self.my_E1)
        self.latent_layer = LatentLayer()
        self.prediction_layer = PredictionLayer()

        
    def call(self, inputs, return_latent_space = False, debug = True):
        x = self.feature_mapping_layer(inputs)
        h = self.lstm_layer(x)
        alpha = self.temporal_attention_layer(h)
        e = self.latent_layer([alpha,h])
        y_pred = self.prediction_layer(e)
        if debug: 
            return {
                "x":x, 
                "h": h, 
                "alpha": alpha, 
                "e": e, 
                "y_pred": y_pred
            }
        if return_latent_space: return e, y_pred
        return y_pred

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape(persistent=True) as tape:
            e, y_pred = self(x, training=True, return_latent_space = True)
            loss1 = tf.keras.losses.hinge(y, y_pred)

            g = tape.gradient(loss1, e)
            e_adv = e + self.my_beta * tf.nn.l2_normalize(g)
            y_pred_adv = self.prediction_layer(e_adv)

            loss = loss1 + tf.keras.losses.hinge(y, y_pred_adv) + sum(self.losses)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        del tape
        return self.compute_metrics(x, y, y_pred, sample_weight)

threshold = 0.5  

def mcc_metric(y_true, y_pred):
  predicted = tf.cast(tf.greater(y_pred, threshold), tf.float32)
  true_pos = tf.math.count_nonzero(predicted * y_true)
  true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
  false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
  false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
  x = tf.cast((true_pos + false_pos) * (true_pos + false_neg) 
      * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
  return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)


def main():
    bs = 64
    T = 10
    D = 12 #nb feature de base
    model = AdvALSTMModel(
        my_beta = 0.05, 
        my_epsilon = 0.01, 
        my_L2 = 0.1, 
        my_E = 16, 
        my_U = 16, 
        my_E1 = 16
        )
    model.compile(
        optimizer = "adam",
        metrics = ["acc"]
    )
    inputs = tf.random.normal(shape=(bs, T, D))

    print(model(inputs))

if __name__ == '__main__':
    main()