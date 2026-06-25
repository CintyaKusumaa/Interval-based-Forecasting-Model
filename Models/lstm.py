# PREPARE DATA
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

low_series   = pd.Series(df["Low"].values,   index=df["Date"], name="Low")
high_series  = pd.Series(df["High"].values,  index=df["Date"], name="High")
open_series  = pd.Series(df["Open"].values,  index=df["Date"], name="Open")
close_series = pd.Series(df["Close"].values, index=df["Date"], name="Close")

# BUILD SEQUENCES
def make_sequences(values, lookback):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

# LSTM MODEL BUILDER
def build_lstm_model(hp, lookback):
    model = models.Sequential()
    model.add(layers.Input(shape=(lookback, 1)))

    units_1 = hp.Int("units_1", min_value=16, max_value=128, step=16)
    dropout_1 = hp.Float("dropout_1", min_value=0.0, max_value=0.4, step=0.1)
    num_layers = hp.Int("num_layers", min_value=1, max_value=2, step=1)

    model.add(layers.LSTM(units_1, return_sequences=(num_layers == 2)))
    model.add(layers.Dropout(dropout_1))

    if num_layers == 2:
        units_2 = hp.Int("units_2", min_value=16, max_value=128, step=16)
        dropout_2 = hp.Float("dropout_2", min_value=0.0, max_value=0.4, step=0.1)
        model.add(layers.LSTM(units_2))
        model.add(layers.Dropout(dropout_2))

    dense_units = hp.Int("dense_units", min_value=8, max_value=64, step=8)
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(1))

    lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 5e-4, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model

# MAIN PIPELINE
def rolling_lstm(
    level_series, name="series",
    train_ratio=0.8, lookback_candidates=(5, 10, 20),
    max_trials=8, tune_epochs=20,
    final_epochs=25, batch_size=16
):
    # split
    train_series, test_series = split_series(level_series, train_ratio=train_ratio)

    # scaling on train only
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()

    # Bayesian tuning
    best_score = np.inf
    best_lookback = None
    best_hp_values = None

    for lookback in lookback_candidates:
        if len(train_scaled) <= lookback + 10:
            continue
        X_train, y_train = make_sequences(train_scaled, lookback)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: build_lstm_model(hp, lookback),
            objective="val_loss",
            max_trials=max_trials,
            overwrite=True,
            directory="bayes_lstm_dir",
            project_name=f"lstm_norefit_{name}_{lookback}"
        )
        es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True)
        tuner.search(
            X_train, y_train,
            validation_split=0.2,
            epochs=tune_epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]
        val_start = int(len(X_train) * 0.8)
        val_loss = best_model.evaluate(
            X_train[val_start:],
            y_train[val_start:],
            verbose=0
        )[0]
        if val_loss < best_score:
            best_score = val_loss
            best_lookback = lookback
            best_hp_values = best_hp.values

    print(f"[{name}] Best lookback: {best_lookback}")
    print(f"[{name}] Best hyperparameters: {best_hp_values}")

    # final fit on train
    lookback = best_lookback
    X_train, y_train = make_sequences(train_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    hp_fixed = kt.HyperParameters()
    for k, v in best_hp_values.items():
        hp_fixed.values[k] = v
    final_model = build_lstm_model(hp_fixed, lookback)
    es_final = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True)
    final_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=final_epochs,
        batch_size=batch_size,
        callbacks=[es_final],
        verbose=0)

    # in-sample fitted
    fitted_scaled = final_model.predict(X_train, verbose=0).flatten()
    fitted_price = scaler.inverse_transform(fitted_scaled.reshape(-1, 1)).flatten()
    insample_pred = pd.Series(
        fitted_price,
        index=train_series.index[lookback:],
        name=f"insample_pred_{name}"
    )

    # rolling forecast 
    all_scaled = scaler.transform(level_series.values.reshape(-1, 1)).flatten()
    train_len = len(train_series)
    forecast_list = []

    for i in range(len(test_series)):
        end_pos = train_len + i
        start_pos = end_pos - lookback
        x_input = all_scaled[start_pos:end_pos].reshape(1, lookback, 1)
        pred_scaled = final_model.predict(x_input, verbose=0).flatten()[0]
        pred_price = scaler.inverse_transform(np.array([[pred_scaled]])).flatten()[0]
        forecast_list.append(pred_price)
    forecast_df = pd.DataFrame({
        "actual_price": test_series,
        "forecast_price": pd.Series(forecast_list, index=test_series.index),
        "naive_price": level_series.shift(1).loc[test_series.index]
    }, index=test_series.index)
    return {
        "name": name,
        "train_series": train_series,
        "test_series": test_series,
        "best_lookback": best_lookback,
        "best_hp": best_hp_values,
        "insample_pred": insample_pred,
        "forecast_df": forecast_df,
        "model": final_model,
        "scaler": scaler}

# RUN ALL SERIES
res_lstm_low = rolling_lstm(level_series=low_series, name="low", train_ratio=0.8,lookback_candidates=(5, 10, 20, 50), max_trials=10, tune_epochs=20,batch_size=16)

res_lstm_high = rolling_lstm(level_series=high_series, name="high", train_ratio=0.8, lookback_candidates=(5, 10, 20, 50), max_trials=10, tune_epochs=20, batch_size=16)

res_lstm_close = rolling_lstm(level_series=close_series, name="close", train_ratio=0.8, lookback_candidates=(5, 10, 20, 50), max_trials=10, tune_epochs=20, batch_size=16)
