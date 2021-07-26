from tensorflow.keras import layers
import tensorflow as tf
 
class GATv2SelfAttention(layers.Layer):
    def __init__(self, seq_size=8, num_heads=8, ff_dim=2048, coef_drop=0.1, in_drop=0.2, alpha=0.3, residual=True, activation=lambda x: x):
        super(GATv2SelfAttention, self).__init__()
        self.heads = []
        self.num_heads = num_heads
        self.alpha = alpha
        self.residual = residual
        self.coef_drop = coef_drop
        self.in_drop = in_drop
        self.seq_size = seq_size
        self.ff_dim = ff_dim
        self.activation = activation
        self.head_weights = []

    def build(self, input_shape):
        for i in range(self.num_heads):
            self.head_weights.append({
                'att_weights':self.add_weight(f'att_weights_{i}',
                                            shape=(input_shape[2],1),
                                            initializer=tf.keras.initializers.GlorotNormal(),
                                            trainable=True),
                'W':self.add_weight(f'W_{i}',
                                    shape=(2*input_shape[2],input_shape[2]),
                                    initializer=tf.keras.initializers.GlorotNormal(),
                                    trainable=True),
                'bias':self.add_weight(f'bias_{i}',
                                        shape=(input_shape[2],),
                                        initializer='zeros',
                                        trainable=True)
            })
            self.heads.append({
                'coef_drop':layers.Dropout(self.coef_drop),
                'in_drop':layers.Dropout(self.in_drop),
                'leaky_relu':layers.LeakyReLU(self.alpha)
            })
        super(GATv2SelfAttention, self).build(input_shape)

    def call(self, inputs, training):
        # GATv2 https://arxiv.org/pdf/2105.14491.pdf

        # The first element of the sequence is considered h_i
        # Concatenate sequence (for each N(h_i) : [h_i || h_j])
        h = inputs
        h_chunk = tf.stack([h[:,0,...]]*self.seq_size,axis=1)
        h_cat = tf.concat([h_chunk,h],axis=-1)

        attns = None
        for i in range(self.num_heads):
            
            # Linear on concatenated sequence
            Wh = tf.matmul(h_cat, self.head_weights[i]['W'])
            Wh = self.heads[i]['leaky_relu'](Wh) # (bs, seq, ff_dim) 

            # Self-Attention outside leaky relu
            e = tf.squeeze(tf.matmul(Wh,self.head_weights[i]['att_weights']),axis=-1) # (bs,seq)
            
            attention = tf.nn.softmax(e,axis=1)
            attention = self.heads[i]['coef_drop'](attention)

            # Broadcasting for keeping dimensions for residual sum
            attention = tf.reshape(tf.repeat(attention, repeats=[self.ff_dim]*self.seq_size,axis=1),shape=(tf.shape(attention)[0],self.seq_size,self.ff_dim))
            
            # Applying attention
            Wh = self.heads[i]['in_drop'](Wh)
            att_out = attention * Wh
            att_out = att_out + self.head_weights[i]['bias']

            # Residual sum and activation
            if self.residual:
                att_out = self.activation(att_out+inputs) # (bs,seq,ff_dim)
            else:
                att_out = self.activation(att_out)

            if attns == None:
                attns = att_out
            else: attns += att_out
        return attns/self.num_heads
