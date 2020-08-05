import os
import functools
import tensorflow as tf
import Music_generation_with_RNN.preprocess

def generate_text(model, char2idx, idx2char, start_string, generation_length=1000):

  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Here batch size == 1
  model.reset_states()
  for i in range(generation_length):
      predictions = model(input_eval) # TODO
      
      # Remove the batch dimension
      predictions = tf.squeeze(predictions, 0)
      
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy() # TODO 
      
      # Pass the prediction along with the previous hidden state
          # as the next inputs to the model
      input_eval = tf.expand_dims([predicted_id], 0)
      
      # Hint: consider what format the prediction is in, vs. the output
      text_generated.append(idx2char[predicted_id]) # TODO 

  return (start_string + ''.join(text_generated))

def main():
    '''
        Load saved model

    '''
    text = generate_text(model, start_string="X")