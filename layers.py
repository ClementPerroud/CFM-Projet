import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, dropout = 0):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.dropout = dropout
    
    def build(self, input_shape):
        self.gru = tf.keras.layers.GRU(self.units, return_sequences = True, dropout=self.dropout)
        self.W1 = self.add_weight(name='W1', shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.V = self.add_weight(name='V', shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        # inputs shape == (batch_size, seq_len, hidden_dim)
        hidden_states = self.gru(inputs)

        # hidden_states shape == (batch_size, seq_len, hidden_dim)
        # score shape == (batch_size, seq_len, units)
        score = tf.nn.tanh(tf.matmul(hidden_states, self.W1))

        # attention_weights shape == (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(tf.matmul(score, tf.expand_dims(self.V, axis=1)), axis=1)

        # context_vector shape after sum == (batch_size, units)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)

        return context_vector