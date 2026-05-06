import pandas as pd
import tensorflow as tf

class HistoryCSV(tf.keras.callbacks.Callback):
    def __init__(self, filepath="training_history.csv", save_every=5):
        super().__init__()
        self.filepath = filepath
        self.save_every = save_every
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["epoch"] = epoch

        # Get current learning rate
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            # For schedules: evaluate at current step
            logs["lr"] = float(lr(self.model.optimizer.iterations))
        else:
            logs["lr"] = float(lr)

        self.history.append(logs.copy())

        if (epoch + 1) % self.save_every == 0:
            pd.DataFrame(self.history).to_csv(self.filepath, index=False)
            print(f"\nHistory saved to {self.filepath}")

    def on_train_end(self, logs=None):
        pd.DataFrame(self.history).to_csv(self.filepath, index=False)