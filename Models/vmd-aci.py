# UTILITAS DASAR
def inverse_diff(pred_diff, last_value):
    """
    Mengembalikan prediksi difference menjadi level interval.
    pred_diff : shape (H, 2)
    last_value: shape (2,)
    """
    preds = []
    prev = np.asarray(last_value, dtype=float).copy()
    for diff in pred_diff:
        x = prev + diff
        preds.append(x)
        prev = x
    return np.asarray(preds, dtype=float)
    
def hukuhara_diff(A, B):
    """
    Hukuhara difference sederhana:
    A, B shape (n, 2)
    """
    return np.column_stack([A[:, 0] - B[:, 0], A[:, 1] - B[:, 1]])

def decompose_interval_vmd(X, K, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    """
    Dekomposisi VMD untuk interval series X shape (T, 2)
    kolom: [left_bound, right_bound]
    
    Output:
    X_modes : list[np.ndarray]
        Jika K == 1: [mode_low_freq, mode_residual]
        Jika K > 1: [mode_1, mode_2, ..., mode_K]
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X harus shape (T, 2).")
    signal_left = X[:, 0]
    signal_right = X[:, 1]

    u_left, _, _ = VMD(signal_left, alpha, tau, K, DC, init, tol)
    u_right, _, _ = VMD(signal_right, alpha, tau, K, DC, init, tol)

    if u_left.shape[0] != K or u_right.shape[0] != K:
        raise ValueError("Output VMD tidak sesuai dengan jumlah mode K.")
    T_hist = min(len(signal_left), u_left.shape[1], u_right.shape[1])

    if K == 1:
        # mode 1 = open-frequency, mode 2 = residual/close-frequency
        u_left_residual = signal_left[:T_hist] - u_left[0][:T_hist]
        u_right_residual = signal_right[:T_hist] - u_right[0][:T_hist]
        X_modes = [
            np.column_stack([u_left[0][:T_hist], u_right[0][:T_hist]]),
            np.column_stack([u_left_residual, u_right_residual]),
        ]
    else:
        X_modes = [
            np.column_stack([u_left[k][:T_hist], u_right[k][:T_hist]])
            for k in range(K)]
    return X_modes

# BUILDER MODEL VMD-ACI
def fit_vmd_aci(
    X_train, K, aci_class, aci_p=1,
    aci_q=2, aci_K=None, alpha=2000, tau=0,
    DC=0, init=1, tol=1e-7,
):
    """
    Semua mode hasil VMD diprediksi oleh ACI.
    Untuk setiap mode, ACI di-fit pada series hasil differencing.
    """
    X_modes = decompose_interval_vmd(
        X_train, K=K, alpha=alpha, tau=tau, DC=DC, init=init, tol=tol
    )
    aci_models = []
    for mode in X_modes:
        if len(mode) < 2:
            raise ValueError("Panjang mode terlalu pendek untuk ACI.")
        Y_mode = hukuhara_diff(mode[1:], mode[:-1])

        if aci_K is None:
            model = aci_class(p=aci_p, q=aci_q)
        else:
            model = aci_class(p=aci_p, q=aci_q, K=aci_K)
        model.fit(Y_mode)
        aci_models.append(model)
    return {
        "type": "VMD-ACI",
        "K": K,
        "models": aci_models,
        "n_modes": len(X_modes),
        "alpha": alpha,
        "tau": tau,
        "DC": DC,
        "init": init,
        "tol": tol,
    }

# ROLLING FORECAST VMD-ACI
def rolling_forecast_vmd_aci(X_train, X_test, fitted_obj):
    """
    Rolling forecast:
    - history di-update pakai data aktual test
    - dekomposisi VMD dilakukan ulang pada history tiap step
    - semua mode diprediksi ACI
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
            if len(X_k) < 2:
                raise ValueError(f"Mode-{k+1} terlalu pendek untuk differencing ACI.")
            X_diff = hukuhara_diff(X_k[1:], X_k[:-1])
            fc_diff = np.asarray(model.forecast_1step(X_diff), dtype=float)
            if fc_diff.shape != (2,):
                raise ValueError(f"Output ACI mode-{k+1} harus shape (2,), dapat {fc_diff.shape}")
            fc_k = inverse_diff(fc_diff.reshape(1, 2), X_k[-1])[0]
            fc_modes.append(fc_k)
            preds_modes[k].append(fc_k)
        fc_final = np.sum(np.stack(fc_modes, axis=0), axis=0)
        preds_final.append(fc_final)

        # rolling update pakai data aktual
        history_raw = np.vstack([history_raw, X_test[t]])

    return np.asarray(preds_final), [np.asarray(p) for p in preds_modes]

# MAIN PAGE
base_dir = "/kaggle/working/VMD-ACI"
os.makedirs(base_dir, exist_ok=True)

results_vmd_aci = {}
res = []
metrics = {}
for K in range(1, 6):
    print(f"Running VMD-ACI with K={K}...")
    # 1. FIT MODEL
    fitted = fit_vmd_aci(
        X_train=X_train,
        K=K,
        aci_class=ACI,
        aci_p=1,
        aci_q=2,
        aci_K=K_scm,   # atau None
        alpha=2000,
        tau=0,
        DC=0,
        init=1,
        tol=1e-7
    )

    # 2. FORECAST
    preds, preds_mode = rolling_forecast_vmd_aci(
        X_train=X_train,
        X_test=X_test,
        fitted_obj=fitted
    )

    # 3. BUAT FOLDER KHUSUS PER K
    k_dir = os.path.join(base_dir, f"K_{K}")
    models_dir = os.path.join(k_dir, "models")
    os.makedirs(k_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 4. SIMPAN MODEL PER MODE
    for i, model in enumerate(fitted["models"], start=1):
        model_path = os.path.join(models_dir, f"aci_mode_{i}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # optional: simpan object fitted lengkap
    fitted_path = os.path.join(k_dir, f"fitted_vmd_aci_K{K}.pkl")
    with open(fitted_path, "wb") as f:
        pickle.dump(fitted, f)

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

    # 7. SIMPAN KE DICTIONARY JUGA
    results_vmd_aci[K] = {
        "fitted": fitted,
        "preds": preds,
        "preds_mode": preds_mode,
        "folder": k_dir
    }

    # 8. EVALUASI
    metrics = evaluate_interval_forecast_zheng(
        preds_sub=preds,      # shape (n, 2)
        actual_L_sub=real_L,  # endpoint kiri
        actual_R_sub=real_H   # endpoint kanan
    )
    metrics["K"] = K
    res.append(metrics)

print("Selesai. Semua model dan output tersimpan di /kaggle/working/output_vmd_aci")
