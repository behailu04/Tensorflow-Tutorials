import tensorflow as tf
import os
import functools
import numpy as np

'''
        Dataset analysis
    Dataset: Thousands of Irish folk songs 
'''
# import util as util

# Inspect dataset

def data_inspect(file_path):
    text = open(file_path).read()
    print('Length of text: {} characters'.format(len(text)))
    
    # util.play_generated_song(text)
    print(text[:250])
    
    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))

    return text, vocab

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def data_preprocess(vocab, text):
    '''
        Vectorize the text : convert the characters to numbers & map the
                            numbers back to characters.

    '''
    char2idx = {u: i for i, u in enumerate(vocab)}
    text_as_int = np.array([char2idx[c] for c in text])
    idx2char = np.array(vocab)
    print('{')
    for char,_ in zip(char2idx, range(20)):
        print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    print('  ...\n}')

    '''
        Creating training examples & targets

    '''
    seq_length = 100
    examples_per_epoch = len(text)//seq_length

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)
    for input_example, target_example in dataset.take(1):
        for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
            print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
    # Batch size 
    BATCH_SIZE = 64
    steps_per_epoch = examples_per_epoch//BATCH_SIZE

    # Buffer size is similar to a queue size
    # This defines a manageable data size to put into memory, where elements are shuffled
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    return dataset



def main():
    file_path = tf.keras.utils.get_file('irish.abc', './data')
    text, vocab  = data_inspect(file_path)
    dataset = data_preprocess(vocab, text)
    return dataset, text, vocab

if __name__ == "__main__" :
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
    main()