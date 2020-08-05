import tensorflow as tf 
import numpy as np 
import os
import time
import functools
import preprocess

# import util as util

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    lstm_layer = tf.keras.layers.LSTM(rnn_units, recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform',
                                return_sequences=True, stateful=True)
    embeding_layer = tf.keras.layers.Embedding(vocab_size, 
                                embedding_dim, batch_input_shape=[batch_size, None])
    
    dense_layer = tf.keras.layers.Dense(vocab_size)

    model = tf.keras.Sequential([embeding_layer, lstm_layer, dense_layer])

    return model

def compute_loss(labels, logits):
    return tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def train(model, dataset, checkpoint_dir):
    EPOCHS = 5

    optimizer = tf.keras.optimizers.Adam
    
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    history = []
    
    for epoch in range(EPOCHS):
        hidden = model.reset_states()

        for inp, target in dataset:
            with tf.GradientTape() as tape:
                predicitions = model(inp)
                loss = compute_loss(target, predicitions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=grads, self=)

            history.append(loss.numpy().mean())
        
        model.save_weights(checkpoint_prefix.format(epoch=epoch))



def main():
    dataset, text, vocab = preprocess.main()
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024
    
    checkpoint_dir = './training_checkpoints'
    
    seq_length = 100
    examples_per_epoch = len(text)//seq_length
    
    BATCH_SIZE = 64
    steps_per_epoch = examples_per_epoch//BATCH_SIZE

    # Buffer size is similar to a queue size
    # This defines a manageable data size to put into memory, where elements are shuffled
    BUFFER_SIZE = 10000

    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
    train(model, dataset, checkpoint_dir)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    model.summary() 

if __name__ == "__main__":
    main()