import tensorflow as tf
from transformers import BertTokenizer
from modules.Transformer import Transformer

tokenizers = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=20):

    sentence = self.tokenizers.encode(sentence,  add_special_tokens = False, padding="max_length", max_length=256,truncation=True, return_tensors="tf")

    encoder_input = sentence

    # as the target is english, the first token to the transformer should be the
    # english start token.
    start = tf.constant([101], dtype=tf.int64)
    end = tf.constant([102], dtype=tf.int64)

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output])

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.squeeze(tf.transpose(output_array.stack()))
    # output.shape (1, tokens)
    text = tokenizers.decode(output)  # shape: ()

    return text
  
def print_translation(sentence, tokens):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens}')
    
if __name__ == "__main__":
    transformer = Transformer(256 , 512, 4,
                            4, 119547, dropout=0.1)

    ts = Translator(tokenizers, transformer)

    sentence = "안녕 나는 트랜스포머야"
    
    text = ts(sentence)
    print_translation(sentence,text)