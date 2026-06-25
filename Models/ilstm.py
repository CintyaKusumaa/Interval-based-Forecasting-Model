from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

@dataclass
class iLSTMConfig:
    lookback: int = 250           # Zheng: 6000 (kalau data cukup panjang)
    n_features: int = 2           # interval endpoints [L, R]
    hidden_units: int = 50        # Zheng text: 50; Table: 20
    learning_rate: float = 1e-3   # Zheng: 0.001
    batch_size: int = 32
    epochs: int = 50
    val_split: float = 0.1
    patience: int = 8
    normalize: bool = True        # normalisasi input (recommended)

    # pilihan loss:
    # "mse", "mae", "huber"
    # atau langsung tf.keras loss object
    loss: Union[str, tf.keras.losses.Loss] = "huber"

    # dipakai kalau loss="huber"
    huber_delta: float = 1.0

class iLSTM:
    """
    Interval LSTM (iLSTM) for predicting next-step interval endpoints [D_L, D_R].

    Input:  sequence of shape (lookback, 2)
    Output: vector of shape (2,)

    Gate mechanism (forget/input/output/cell) is inherent in Keras LSTM.
    """

    def __init__(self, config: iLSTMConfig):
        self.cfg = config
        self.model: tf.keras.Model | None = None
        self.mu: np.ndarray | None = None  # shape (1,1,2)
        self.sd: np.ndarray | None = None  # shape (1,1,2)

    def _get_loss(self):
        """Resolve loss from config."""
        loss = self.cfg.loss

        if isinstance(loss, tf.keras.losses.Loss):
            return loss

        if isinstance(loss, str):
            loss = loss.lower()

            if loss == "mse":
                return tf.keras.losses.MeanSquaredError()
            elif loss == "mae":
                return tf.keras.losses.MeanAbsoluteError()
            elif loss == "huber":
                return tf.keras.losses.Huber(delta=self.cfg.huber_delta)
            else:
                raise ValueError(
                    f"Unknown loss '{self.cfg.loss}'. "
                    "Use 'mse', 'mae', 'huber', or a tf.keras loss object."
                )

        raise TypeError(
            "cfg.loss must be a string ('mse', 'mae', 'huber') "
            "or a tf.keras.losses.Loss object."
        )
   
    def _build_model(self) -> tf.keras.Model:
        inp = layers.Input(shape=(self.cfg.lookback, self.cfg.n_features),name="input_interval_seq")
        x = layers.LSTM(
            units=self.cfg.hidden_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=False,
            name=f"lstm_hidden_{self.cfg.hidden_units}"
        )(inp)
        out = layers.Dense(2, name="output_interval")(x)

        m = models.Model(inp, out, name="iLSTM_ZhengStyle")
        m.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = self.cfg.learning_rate), 
            loss=self._get_loss()
        )
        return m

    @staticmethod
    def _make_sequences(X: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"X must have shape (T,2), got {X.shape}")
        if len(X) <= lookback:
            raise ValueError(f"Not enough data: len(X)={len(X)} <= lookback={lookback}")
        Xs, ys = [], []
        for t in range(lookback, len(X)):
            Xs.append(X[t - lookback:t, :])
            ys.append(X[t, :])
        return np.array(Xs), np.array(ys)

    def _fit_normalizer(self, Xseq: np.ndarray):
        # Xseq: (N, lookback, 2)
        self.mu = Xseq.mean(axis=(0, 1), keepdims=True)
        self.sd = Xseq.std(axis=(0, 1), keepdims=True) + 1e-8

    def _norm_X(self, Xseq: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize:
            return Xseq
        if self.mu is None or self.sd is None:
            raise RuntimeError("Normalizer not fitted.")
        return (Xseq - self.mu) / self.sd

    def _norm_y(self, y: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize:
            return y
        if self.mu is None or self.sd is None:
            raise RuntimeError("Normalizer not fitted.")
        return (y - self.mu.reshape(1, 2)) / self.sd.reshape(1, 2)

    def _denorm_y(self, yhat: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize:
            return yhat
        return yhat * self.sd.reshape(2,) + self.mu.reshape(2,)

    def fit(self, X_interval: np.ndarray, verbose: int = 1) -> "iLSTM":
        """
        Fit iLSTM on interval series X_interval (T,2).
        """
        Xseq, yseq = self._make_sequences(X_interval, self.cfg.lookback)
        if self.cfg.normalize:
            self._fit_normalizer(Xseq)
            Xseq = self._norm_X(Xseq)
            yseq = self._norm_y(yseq)
        self.model = self._build_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.patience,
                min_delta=1e-5,
                mode="min",
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )
        ]
        self.model.fit(
            Xseq, yseq,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            validation_split=self.cfg.val_split,
            callbacks=callbacks,
            verbose=verbose
        )
        return self

    def forecast_1step(self, X_history: np.ndarray) -> np.ndarray:
        """
        Forecast next interval endpoints [D_L, D_R] given history (T,2).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_history = np.asarray(X_history, dtype=float)
        if len(X_history) < self.cfg.lookback:
            raise ValueError(f"Need at least lookback={self.cfg.lookback} history points.")
        x = X_history[-self.cfg.lookback:, :][None, :, :]  # (1,lookback,2)
        x = self._norm_X(x) if self.cfg.normalize else x
        yhat = self.model.predict(x, verbose=0)[0]  # (2,)
        yhat = self._denorm_y(yhat)
        return yhat

    def save(self, path: str):
        """
        Save keras model + normalizer.
        Creates:
          - path (SavedModel)
          - path + ".npz" (mu/sd/config)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        self.model.save(path)

        np.savez(
            path + ".npz",
            mu=self.mu,
            sd=self.sd,
            lookback=self.cfg.lookback,
            n_features=self.cfg.n_features,
            hidden_units=self.cfg.hidden_units,
            learning_rate=self.cfg.learning_rate,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epochs,
            val_split=self.cfg.val_split,
            patience=self.cfg.patience,
            normalize=int(self.cfg.normalize),
        )

    @classmethod
    def load(cls, path: str) -> "iLSTM":
        """
        Load keras model + normalizer/config saved by save().
        """
        data = np.load(path + ".npz", allow_pickle=True)

        cfg = iLSTMConfig(
            lookback=int(data["lookback"]),
            n_features=int(data["n_features"]),
            hidden_units=int(data["hidden_units"]),
            learning_rate=float(data["learning_rate"]),
            batch_size=int(data["batch_size"]),
            epochs=int(data["epochs"]),
            val_split=float(data["val_split"]),
            patience=int(data["patience"]),
            normalize=bool(int(data["normalize"])),
        )

        obj = cls(cfg)
        obj.model = tf.keras.models.load_model(path)
        obj.mu = data["mu"] if "mu" in data else None
        obj.sd = data["sd"] if "sd" in data else None
        return obj
