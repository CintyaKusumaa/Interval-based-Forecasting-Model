# TUNING HYPERPARAMETER iLSTM DENGAN OPTUNA
def tune_pure_ilstm_optuna(
    X_train_full, lookback_choices=(50, 100, 200),
    hidden_units_choices=(16, 32, 50, 64),
    learning_rate_min=1e-4,
    learning_rate_max=1e-3,
    loss_choices=("mse", "mae", "huber"),
    huber_delta_choices=(1.0,),
    val_ratio=0.2, metric="mse",
    n_trials=30, timeout=None,
    sampler_seed=42, verbose=True,
    print_best_only=True, save_dir=None,
    study_name="pure_ilstm_optuna",
    refit_best_on_full_data=True,
    save_csv=True
):
    import os
    import json
    import time
    import joblib
    import optuna
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    X_train_full = np.asarray(X_train_full, dtype=float)
    if X_train_full.ndim != 2 or X_train_full.shape[1] != 2:
        raise ValueError("X_train_full harus shape (T, 2).")
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    n = len(X_train_full)
    split = int((1 - val_ratio) * n)

    train_data = X_train_full[:split]
    val_data = X_train_full[split:]

    if len(train_data) < 5 or len(val_data) < 5:
        raise ValueError("Data train/validation terlalu pendek.")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    trial_logs = []

    def objective(trial):
        trial_start = time.time()

        lookback = trial.suggest_categorical("lookback", list(lookback_choices))
        hidden_units = trial.suggest_categorical("hidden_units", list(hidden_units_choices))
        learning_rate = trial.suggest_float(
            "learning_rate",
            learning_rate_min,
            learning_rate_max,
            log=True
        )
        loss_name = trial.suggest_categorical("loss", list(loss_choices))

        if loss_name == "huber":
            huber_delta = trial.suggest_categorical("huber_delta", list(huber_delta_choices))
        else:
            huber_delta = None

        trial_info = {
            "trial_number": trial.number,
            "lookback": lookback,
            "hidden_units": hidden_units,
            "learning_rate": float(learning_rate),
            "loss": loss_name,
            "huber_delta": None if huber_delta is None else float(huber_delta),
            "metric": metric,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "train_loss_last": np.nan,
            "val_loss_last": np.nan,
            "best_train_loss": np.nan,
            "best_val_loss": np.nan,
            "n_epochs_ran": np.nan,
            "forecast_score": np.nan,
            "fit_runtime_sec": np.nan,
            "trial_runtime_sec": np.nan,
            "status": "FAILED",
            "error_message": None
        }

        # validasi lookback
        if lookback >= len(train_data) or lookback >= len(val_data):
            trial_info["status"] = "PRUNED_LOOKBACK"
            trial_info["error_message"] = "lookback >= len(train_data) atau len(val_data)"
            trial_info["trial_runtime_sec"] = time.time() - trial_start
            trial_logs.append(trial_info)

            for k, v in trial_info.items():
                if v is None or isinstance(v, (int, float, str)):
                    trial.set_user_attr(k, v)

            raise optuna.exceptions.TrialPruned()

        cfg = iLSTMConfig(
            lookback=lookback,
            hidden_units=hidden_units,
            learning_rate=learning_rate,
            loss=loss_name,
            huber_delta=1.0 if huber_delta is None else huber_delta
        )

        try:
            fit_start = time.time()
            # IMPORTANT:
            # pastikan iLSTM.fit() mendukung argumen verbose
            # dan diteruskan ke model.fit(..., verbose=verbose)
            model = iLSTM(cfg).fit(train_data, verbose=0 if verbose else 0)
            fit_runtime = time.time() - fit_start
            trial_info["fit_runtime_sec"] = fit_runtime

            # ambil history training kalau tersedia
            hist_dict = None
            if hasattr(model, "history") and model.history is not None:
                if hasattr(model.history, "history"):
                    hist_dict = model.history.history
                elif isinstance(model.history, dict):
                    hist_dict = model.history

            elif hasattr(model, "model") and hasattr(model.model, "history") and model.model.history is not None:
                if hasattr(model.model.history, "history"):
                    hist_dict = model.model.history.history

            if hist_dict is not None:
                train_losses = hist_dict.get("loss", [])
                val_losses = hist_dict.get("val_loss", [])
                if len(train_losses) > 0:
                    trial_info["train_loss_last"] = float(train_losses[-1])
                    trial_info["best_train_loss"] = float(np.min(train_losses))
                    trial_info["n_epochs_ran"] = len(train_losses)
                if len(val_losses) > 0:
                    trial_info["val_loss_last"] = float(val_losses[-1])
                    trial_info["best_val_loss"] = float(np.min(val_losses))

            # rolling forecast validation
            history = train_data.copy()
            preds = []
            for t in range(len(val_data)):
                fc = model.forecast_1step(history)
                preds.append(fc)
                history = np.vstack([history, val_data[t]])
            preds = np.asarray(preds, dtype=float)
            if metric == "mse":
                score = mean_squared_error(val_data, preds)
            elif metric == "mae":
                score = mean_absolute_error(val_data, preds)
            else:
                raise ValueError("metric harus 'mse' atau 'mae'")

            trial_info["forecast_score"] = float(score)
            trial_info["status"] = "SUCCESS"
            trial_info["trial_runtime_sec"] = time.time() - trial_start
            for k, v in trial_info.items():
                if v is None or isinstance(v, (int, float, str)):
                    trial.set_user_attr(k, v)

            trial_logs.append(trial_info)
            return float(score)
        except Exception as e:
            trial_info["status"] = "FAILED"
            trial_info["error_message"] = str(e)
            trial_info["trial_runtime_sec"] = time.time() - trial_start
            for k, v in trial_info.items():
                if v is None or isinstance(v, (int, float, str)):
                    trial.set_user_attr(k, v)
            trial_logs.append(trial_info)
            raise optuna.exceptions.TrialPruned()

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name)
    def best_trial_callback(study, trial):
        if not verbose or not print_best_only:
            return
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if study.best_trial.number == trial.number:
            print(
                f"\nBest trial so far: {trial.number} | "
                f"score={trial.value:.6f} | "
                f"params={trial.params}"
            )

    callbacks = [best_trial_callback] if (verbose and print_best_only) else None
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose,
        callbacks=callbacks
    )
    results = []
    for tr in study.trials:
        if tr.value is None:
            continue
        results.append({
            "trial_number": tr.number,
            "lookback": tr.params.get("lookback"),
            "hidden_units": tr.params.get("hidden_units"),
            "learning_rate": tr.params.get("learning_rate"),
            "loss": tr.params.get("loss"),
            "huber_delta": tr.params.get("huber_delta", None),
            "score": tr.value,
            "state": str(tr.state)
        })

    if len(results) == 0:
        raise RuntimeError("Tidak ada trial yang berhasil.")
    results = sorted(results, key=lambda x: x["score"])
    best_cfg = results[0].copy()
    final_train_data = X_train_full if refit_best_on_full_data else train_data

    best_model_cfg = iLSTMConfig(
        lookback=best_cfg["lookback"],
        hidden_units=best_cfg["hidden_units"],
        learning_rate=best_cfg["learning_rate"],
        loss=best_cfg["loss"],
        huber_delta=1.0 if best_cfg["huber_delta"] is None else best_cfg["huber_delta"]
    )
    best_model = iLSTM(best_model_cfg).fit(
        final_train_data,
        verbose=0 if verbose else 0
    )

    tuning_df = pd.DataFrame(trial_logs)
    if "forecast_score" in tuning_df.columns:
        tuning_df = tuning_df.sort_values(
            by=["status", "forecast_score"],
            ascending=[True, True],
            na_position="last"
        ).reset_index(drop=True)
    if save_dir is not None and save_csv:
        csv_path = os.path.join(save_dir, f"{study_name}_trial_results.csv")
        tuning_df.to_csv(csv_path, index=False)

        best_cfg_path = os.path.join(save_dir, f"{study_name}_best_config.json")
        with open(best_cfg_path, "w", encoding="utf-8") as f:
            json.dump(best_cfg, f, indent=2, ensure_ascii=False)
            
        study_path = os.path.join(save_dir, f"{study_name}_study.pkl")
        joblib.dump(study, study_path)

    if verbose:
        print("\nFinal best trial:")
        print(f"  Trial number : {study.best_trial.number}")
        print(f"  Best score   : {study.best_value:.6f}")
        print(f"  Best params  : {study.best_trial.params}")
    return best_cfg, best_model, results, study, tuning_df

# BUILDER MODEL VMD-iLSTM
def fit_vmd_ilstm(
    X_train, K, ilstm_class,
    ilstm_cfg=None,
    alpha=2000, tau=0,
    DC=0, init=1,
    tol=1e-7, use_tuning=False,
    n_trials=15, metric="mae",
    tuning_base_dir="/kaggle/working/VMD-iLSTM/iLSTM_tuning"
):
    """
    Semua mode hasil VMD diprediksi oleh iLSTM.
    Jika use_tuning=True, maka tiap mode akan dituning dulu dengan Optuna.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_modes = decompose_interval_vmd(
        X_train, K=K, alpha=alpha, tau=tau, DC=DC, init=init, tol=tol
    )

    ilstm_models = []
    best_cfgs = []
    tuning_tables = []

    for i, mode in enumerate(X_modes, start=1):
        mode = np.asarray(mode, dtype=float)
        if use_tuning:
            save_dir_mode = os.path.join(tuning_base_dir, f"K_{K}", f"mode_{i}")
            os.makedirs(save_dir_mode, exist_ok=True)
            best_cfg, best_model, results, study, tuning_df = tune_pure_ilstm_optuna(
                X_train_full=mode,
                n_trials=n_trials,
                metric=metric,
                save_dir=save_dir_mode,
                study_name=f"ilstm_tuning_K{K}_mode_{i}",
                refit_best_on_full_data=True,
                verbose=True,
                print_best_only=True
            )
            ilstm_models.append(best_model)
            best_cfgs.append(best_cfg)
            tuning_tables.append(tuning_df)
        else:
            if ilstm_cfg is None:
                raise ValueError("ilstm_cfg tidak boleh None jika use_tuning=False")
            model = ilstm_class(ilstm_cfg).fit(mode)
            ilstm_models.append(model)
            best_cfgs.append(ilstm_cfg)
            tuning_tables.append(None)
    return {
        "type": "VMD-iLSTM",
        "K": K,
        "models": ilstm_models,
        "n_modes": len(X_modes),
        "alpha": alpha,
        "tau": tau,
        "DC": DC,
        "init": init,
        "tol": tol,
        "best_cfgs": best_cfgs,
        "tuning_tables": tuning_tables,
        "use_tuning": use_tuning,
    }
    
# ROLLING FORECAST VMD-iLSTM
def rolling_forecast_vmd_ilstm(X_train, X_test, fitted_obj):
    """
    Rolling forecast:
    - history di-update pakai data aktual test
    - dekomposisi VMD dilakukan ulang pada history tiap step
    - semua mode diprediksi iLSTM
    """
    history_raw = np.asarray(X_train, dtype=float).copy()
    X_test = np.asarray(X_test, dtype=float)

    K = fitted_obj["K"]
    models = fitted_obj["models"]

    preds_final = []
    preds_modes = [[] for _ in range(len(models))]

    for t in range(len(X_test)):
        X_modes_hist = decompose_interval_vmd(
            history_raw,
            K=K,
            alpha=fitted_obj["alpha"],
            tau=fitted_obj["tau"],
            DC=fitted_obj["DC"],
            init=fitted_obj["init"],
            tol=fitted_obj["tol"],
        )

        fc_modes = []
        for k, model in enumerate(models):
            X_k = X_modes_hist[k]

            fc_k = np.asarray(model.forecast_1step(X_k), dtype=float)
            if fc_k.shape != (2,):
                raise ValueError(f"Output iLSTM mode-{k+1} harus shape (2,), dapat {fc_k.shape}")

            fc_modes.append(fc_k)
            preds_modes[k].append(fc_k)

        fc_final = np.sum(np.stack(fc_modes, axis=0), axis=0)
        preds_final.append(fc_final)

        # rolling update pakai data aktual
        history_raw = np.vstack([history_raw, X_test[t]])

    return np.asarray(preds_final), [np.asarray(p) for p in preds_modes]

# MAIN PAGE
import os
import json
import numpy as np
import pandas as pd

base_dir = "/kaggle/working/VMD-iLSTM"
os.makedirs(base_dir, exist_ok=True)

results_vmd_ilstm = {}
res = []

for K in range(1, 6):
    print(f"Running VMD-iLSTM with K={K}...")
    # 1. FIT MODEL
    fitted = fit_vmd_ilstm(
        X_train=X_train,
        K=K,
        ilstm_class=iLSTM,
        ilstm_cfg=None,
        use_tuning=True,
        n_trials=15,
        metric="mae"
    )
    
    # 2. FORECAST
    preds, preds_mode = rolling_forecast_vmd_ilstm(
        X_train=X_train,
        X_test=X_test,
        fitted_obj=fitted)

    # 3. BUAT FOLDER KHUSUS PER K
    k_dir = os.path.join(base_dir, f"K_{K}")
    models_dir = os.path.join(k_dir, "models")
    os.makedirs(k_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 4. SIMPAN MODEL PER MODE
    for i, model in enumerate(fitted["models"], start=1):
        model_path = os.path.join(models_dir, f"ilstm_mode_{i}.keras")
        model.model.save(model_path)
    cfgs_meta = [
        {
            "lookback": cfg["lookback"],
            "hidden_units": cfg["hidden_units"],
            "learning_rate": cfg["learning_rate"],
            "loss": cfg["loss"],
            "huber_delta": cfg["huber_delta"],
        }
        for cfg in fitted["best_cfgs"]
    ]
    fitted_meta = {
        "type": fitted["type"],
        "K": fitted["K"],
        "n_modes": fitted["n_modes"],
        "alpha": fitted["alpha"],
        "tau": fitted["tau"],
        "DC": fitted["DC"],
        "init": fitted["init"],
        "tol": fitted["tol"],
        "use_tuning": fitted["use_tuning"],
        "best_cfgs": cfgs_meta}

    meta_path = os.path.join(k_dir, f"fitted_vmd_ilstm_K{K}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(fitted_meta, f, indent=4, ensure_ascii=False)

    # 5. SIMPAN EXCEL PREDIKSI FINAL
    n_pred = len(preds)
    df_final = pd.DataFrame({
        "real_low": X_test[:n_pred, 0],
        "real_high": X_test[:n_pred, 1],
        "pred_low": preds[:, 0],
        "pred_high": preds[:, 1],
    })
    final_xlsx_path = os.path.join(k_dir, f"prediksi_final_K{K}.xlsx")
    with pd.ExcelWriter(final_xlsx_path, engine="openpyxl") as writer:
        df_final.to_excel(writer, sheet_name="Final", index=False)

    # 6. SIMPAN EXCEL PREDIKSI PER MODE
    mode_xlsx_path = os.path.join(k_dir, f"prediksi_mode_K{K}.xlsx")
    with pd.ExcelWriter(mode_xlsx_path, engine="openpyxl") as writer:
        for i, arr in enumerate(preds_mode, start=1):
            df_mode = pd.DataFrame(arr, columns=["pred_low", "pred_high"])
            df_mode.to_excel(writer, sheet_name=f"Mode_{i}", index=False)

    # 7. SIMPAN KE DICTIONARY
    results_vmd_ilstm[K] = {
        "fitted": fitted,
        "preds": preds,
        "preds_mode": preds_mode,
        "folder": k_dir}

    # 8. EVALUASI
    real_L = X_test[:n_pred, 0]
    real_H = X_test[:n_pred, 1]
    real_L_sub = real_L[:n_pred]
    real_H_sub = real_H[:n_pred]
    metrics = evaluate_interval_forecast_zheng(
        preds_sub=preds,
        actual_L_sub=real_L_sub,
        actual_R_sub=real_H_sub
    )
    metrics["K"] = K
    res.append(metrics)

res_df = pd.DataFrame(res)
eval_path = os.path.join(base_dir, "evaluasi_vmd_ilstm.xlsx")
with pd.ExcelWriter(eval_path, engine="openpyxl") as writer:
    res_df.to_excel(writer, sheet_name="Eval", index=False)

print(f"Selesai. Semua model dan output tersimpan di {base_dir}")
