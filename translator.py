import tensorflow as tf
from transformers import BertTokenizer
from modules.Transformer import Transformer

tokenizers = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=256):

    sentence = self.tokenizers.encode(sentence, padding="max_length", max_length=256,truncation=True, return_tensors="tf")

    encoder_input = sentence

    # as the target is english, the first token to the transformer should be the
    # english start token.
    start = tf.constant([[101]], dtype=tf.int64)
    end = tf.constant([[102]], dtype=tf.int64)

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    output = start

    for _ in tf.range(max_length):
      predictions = self.transformer([encoder_input, output])

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output = tf.concat([output, predicted_id], axis=-1)

      if predicted_id == end:
        break

    output = tf.squeeze(output)

    return output
  
def print_translation(sentence, tokens):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens}')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
if __name__ == "__main__":

    gpus = tf.config.experimental.list_logical_devices('GPU')

    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('\n\n Running on multiple GPUs ', [gpu.name for gpu in gpus])




    with strategy.scope():
        transformer = Transformer(256 , 512, 4,
                                4, 119547, dropout=0)

        ts = Translator(tokenizers, transformer)

        ts.transformer.load_weights("1.weights.h5")

        sentence = "안녕 제발 작동 좀 되라"
        
        text = ts(sentence)

        text = tokenizers.decode(text)

        print_translation(sentence,text)