import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics

from sklearn.preprocessing import MinMaxScaler

sigma = 1.


class Atlas(object):

    def __init__(self, d, k):
        self.d = d
        self.k = k
        self.sc = MinMaxScaler(feature_range=(-1,1))

    def fit(self, X, val_rat = .2, intermediate_dim=64, batch_size=100, epochs=200):
        sigma = 1.
        N = X.shape[1] # ambient dimension

        x = Input(shape=(N,))
        h = Dense(intermediate_dim, activation='relu', name='encoder_hidden')(x)

        z_mean1 = Dense(self.d, activation='tanh', name='z_mean1')(h)
        z_mean2 = Dense(self.d, activation='tanh', name='z_mean2')(h)

        p1 = Dense(1, activation='sigmoid', name='prob_chart_1')(h)

        decoder_h1 = Dense(intermediate_dim, activation='relu', name='decoder1_hidden')
        decoder_h2 = Dense(intermediate_dim, activation='relu', name='decoder2_hidden')

        decoder_mean1 = Dense(N, activation='tanh', name='decoder_mean1')
        decoder_mean2 = Dense(N, activation='tanh', name='decoder_mean2')

        h1_decoded = decoder_h1(z_mean1)
        h2_decoded = decoder_h2(z_mean2)

        x_decoded_mean1 = decoder_mean1(h1_decoded)
        x_decoded_mean2 = decoder_mean2(h2_decoded)

        class CustomVariationalLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CustomVariationalLayer, self).__init__(**kwargs)

            def vae_loss(self, x, x_decoded_mean1, x_decoded_mean2):
                rec_loss = (p1/(2*sigma**2)*metrics.mean_squared_error(x, x_decoded_mean1)
                            +(1-p1)/(2*sigma**2)*metrics.mean_squared_error(x, x_decoded_mean2))
                #kl_loss = (p1*(K.log(p1) - K.sum(K.log(z_shape1*(1-K.abs(z_mean1)))))
                #               + (1-p1)*(K.log(1-p1) - K.sum(K.log(z_shape2*(1-K.abs(z_mean2))))))
                kl_loss = 0.

                # regularization
                regu_loss = -p1*K.log(p1) - (1-p1)*K.log(p1)
                #regu_loss = 0.
                return K.mean(rec_loss + .5*kl_loss + 0.*regu_loss) - K.var(p1)
                # should weight prev term by alpha

            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean1 = inputs[1]
                x_decoded_mean2 = inputs[2]
                loss = self.vae_loss(x, x_decoded_mean1, x_decoded_mean2)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x

        y = CustomVariationalLayer(name='loss')([x, x_decoded_mean1, x_decoded_mean2])
        print('here')
        vae = Model(x, y)

        # scale points to between -1 and 1
        stdpts = self.sc.fit_transform(X)
        s = X.shape[0]
        val_index = np.random.randint(0,s,int(.2*s))
        train_index = np.delete(np.arange(s), val_index)

        x_train, x_test = stdpts[train_index], stdpts[val_index]
        vae.compile(optimizer='rmsprop', loss=None)
        vae.fit(x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test,None))

        # create the generators
        decoder_input1 = Input(shape=(1,))
        _h1_decoded = decoder_h1(decoder_input1)
        _x_decoded_mean1 = decoder_mean1(_h1_decoded)
        self.generator1 = Model(decoder_input1, _x_decoded_mean1)

        decoder_input2 = Input(shape=(1,))
        _h2_decoded = decoder_h2(decoder_input2)
        _x_decoded_mean2 = decoder_mean2(_h2_decoded)
        self.generator2 = Model(decoder_input2, _x_decoded_mean2)

        # create the encoder
        self.encoder = Model(x,[z_mean1, z_mean2, p1])

    # gives the probability of which chart a point belongs in
    def chart_probs(self, X):
        return self.encoder.predict(self.sc.fit_transform(X))[2]

    # map some points into latent space
    def encode():
        pass

    def sample_from_latent(n_samples):
        pass
