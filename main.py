import os
import time
import tensorflow as tf
#import urllib
#from kobert_tokenizer import KoBERTTokenizer
#from transformers import BertTokenizer

from modules.Transformer import Transformer
import pandas as pd
import numpy as np
from data import Dataloader

#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
BATCH_SIZE = 16

# x_train_df = ""
# y_train_df = ""
# x_valid_df = ""
# y_valid_df = ""

# train_filelist = os.listdir("data/train")
# valid_filelist = os.listdir("data/valid")

# x_arr = []
# y_arr = []

# for filename in train_filelist:
#     x = pd.read_csv("data/train/"+filename, usecols=["한국어"]).to_numpy()
#     y = pd.read_csv("data/train/"+filename, usecols=["영어"]).to_numpy()
#     for i in x:
#         x_arr.append(tokenizer.encode(i[0],  add_special_tokens = False, padding="max_length", max_length=256,truncation=True, return_tensors="np"))
#     for i in y:
#         y_arr.append(tokenizer.encode(i[0], add_special_tokens = False, padding="max_length", max_length=256,truncation=True, return_tensors="np"))

# x_train_df = np.array(x_arr)
# y_train_df = np.array(y_arr)

# x_arr = []
# y_arr = []

# for filename in valid_filelist:
#     x = pd.read_csv("data/valid/"+filename, usecols=["한국어"]).to_numpy()
#     y = pd.read_csv("data/valid/"+filename, usecols=["영어"]).to_numpy()
#     for i in x:
#         x_arr.append(tokenizer.encode(i[0], add_special_tokens = False, padding="max_length", max_length=256, return_tensors="np"))
#     for i in y:
#         y_arr.append(tokenizer.encode(i[0],  add_special_tokens = False, padding="max_length", max_length=256, return_tensors="np"))

 
# x_valid_df = np.array(x_arr)
# y_valid_df = np.array(y_arr)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=-1))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

def train_step(inp, tar):
    pad_left = tf.constant([[101]], dtype=tf.int64)
    pad_left = tf.repeat(pad_left, BATCH_SIZE, axis=0)
    tar_inp = tar[:, :-1]
    tar_inp = tf.concat([pad_left, tar_inp], axis=-1)
    

    pad_right = tf.constant([[102]], dtype=tf.int64)
    pad_right = tf.repeat(pad_right, BATCH_SIZE, axis=0)
    tar_real = tar[:, 1:]
    tar_real = tf.concat([tar_real, pad_right], axis=-1)

    with tf.GradientTape() as tape:
        predictions = model([inp, tar_inp],
                                    training = True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    acc = accuracy_function(tar_real, predictions)

    return loss, acc

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

@tf.function
def distributed_train_step(inp, tar, i):
    per_replica_losses, acc = strategy.run(train_step, args=(inp, tar))

    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                            axis=None)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_logical_devices('GPU')

    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('\n\n Running on multiple GPUs ', [gpu.name for gpu in gpus])



    with strategy.scope():
        model = Transformer(256 , 512, 4,
                            4, 119547, dropout=0.1)

        # model = tf.keras.models.load_model("model", custom_objects={"metric": metric, "loss_func": loss_func})


        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        learning_rate = CustomSchedule(256)

        optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                            epsilon=1e-9)



        # np.save("x_train.npy", x_train_df)
        # np.save("y_train.npy", y_train_df)
        # np.save("x_valid.npy", x_valid_df)
        # np.save("y_valid.npy", y_valid_df)



        x_train_df = np.load("x_train.npy") 
        y_train_df = np.load("y_train.npy")
        x_valid_df = np.load("x_valid.npy") 
        y_valid_df = np.load("y_valid.npy")


        train_dataloader = Dataloader(x_train_df, y_train_df, BATCH_SIZE)
        vaild_dataloader = Dataloader(x_valid_df, y_valid_df, BATCH_SIZE)

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

        checkpoint_path = "./checkpoints/train"

        ckpt = tf.train.Checkpoint(transformer=model)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

        metrics_names = ['train_loss', 'train_acc']
        
        

        for epoch in range(10):
            progbar = tf.keras.utils.Progbar(train_dataloader.__len__())
            start = time.time()

            # inp -> korean, tar -> english
            for (batch, (inp, tar)) in enumerate(train_dataloader):
                distributed_train_step(inp, tar, batch)
                progbar.update(batch)

                

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    model.save("model.h5")
