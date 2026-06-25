def split_mode_indices(K):
    aci_idx = [0]
    if K == 1:
        ilstm_idx = [1]
    else:
        ilstm_idx = list(range(1, K))

    return aci_idx, ilstm_idx

# PREPARE DATA DAN MODEL
all_results = {}
for K in range(1, 6):
    print("=" * 70)
    print(f"PROCESSING K = {K}")

    # 1) VMD decomposition
    u_low, u_lowhat, omega_low = VMD(signal_low, alpha, tau, K, DC, init, tol)
    u_high, u_highhat, omega_high = VMD(signal_high, alpha, tau, K, DC, init, tol)
    n = u_low.shape[1]
    t = np.arange(n)
    T_hist = min(u_low.shape[1], u_high.shape[1], len(signal_low), len(signal_high))

    # 2) plot hasil VMD
    if K == 1:
        # Plotting hasil VMD K = 1
        fig, axes = plt.subplots(K+1, 1, figsize=(12, 3*(K+1)), sharex=True)
        fig.suptitle(f"VMD Decomposition with K={K} and Alpha={alpha}", fontsize=14)
        axes = np.atleast_1d(axes)
        for i in range(K+1):
            if i == 1:
                axes[i].plot(t, (signal_low[:T_hist] - u_low[0]), linewidth=1.2, label=f"Low Mode {i+1}")
                axes[i].plot(t, (signal_low[:T_hist] - u_high[0]), linewidth=1.2, label=f"High Mode {i+1}")
                axes[i].set_title(f"High Frequency", fontsize=12, fontweight='semibold')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
            else:
                axes[i].plot(t, u_low[i], linewidth=1.2, label=f"Low Mode {i+1}")
                axes[i].plot(t, u_high[i], linewidth=1.2, label=f"High Mode {i+1}")
                axes[i].set_title(f"Mode {i+1} (Low Frequency)", fontsize=12, fontweight='semibold')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(K, 1, figsize=(12, 3 * K), sharex=True)
        fig.suptitle(f"VMD Decomposition with K={K} and Alpha={alpha}", fontsize=14)
        axes = np.atleast_1d(axes)  # supaya aman saat K=1
        for i in range(K):
            axes[i].plot(t, u_low[i], linewidth=1.2, label=f"Low Mode {i+1}")
            axes[i].plot(t, u_high[i], linewidth=1.2, label=f"High Mode {i+1}")
            axes[i].set_title(f"Mode {i+1}", fontsize=12, fontweight='semibold')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    # 3) bentuk X_listK
    if K == 1:
        X_list = [
            np.column_stack([u_low[0][:T_hist], u_high[0][:T_hist]]),   # mode 1: low freq
            np.column_stack([(signal_low[:T_hist] - u_low[0]), (signal_high[:T_hist] - u_high[0])])  # mode 2: residual
        ]   
    else:
        X_list = [
            np.column_stack([u_low[i][:T_hist], u_high[i][:T_hist]])
            for i in range(K)
        ]
    globals()[f"X_list{K}"] = X_list

    # 4) tentukan mode untuk ACI dan iLSTM
    aci_idx, ilstm_idx = split_mode_indices(K)
    print(f"ACI modes   : {[i+1 for i in aci_idx]}")
    print(f"iLSTM modes : {[i+1 for i in ilstm_idx]}")

    # 5) bangun Y untuk mode ACI
    #    Y = Hukuhara diff dari X_list[idx]
    Y_lists = []
    for idx in aci_idx:
        Y_mode = hukuhara_diff(X_list[idx][1:], X_list[idx][:-1])
        Y_lists.append(Y_mode)

    # buat variabel nama sesuai permintaan
    # contoh:
    # K=1  -> Y_list1
    # K=3  -> Y1_list3, Y2_list3
    if len(Y_lists) == 1:
        globals()[f"Y_list{K}"] = Y_lists[0]
    else:
        for j, Y_mode in enumerate(Y_lists, start=1):
            globals()[f"Y{j}_list{K}"] = Y_mode

    # 6) fit model ACI
    aci_models = []
    for j, Y_mode in enumerate(Y_lists, start=1):
        model = ACI(p=1, q=2, K=K_scm)
        model.fit(Y_mode)
        aci_models.append(model)

        # naming sesuai permintaan
        # K=1  -> aci_model1
        # K=3  -> aci1_model3, aci2_model3
        if len(Y_lists) == 1:
            globals()[f"aci_model{K}"] = model
            print(f"\n[ACI model K={K}]")
            print("Params:", model.params_)
            print("obj(init):", model._objective(model.params_, Y_mode))
        else:
            globals()[f"aci{j}_model{K}"] = model
            print(f"\n[ACI model {j} | K={K}]")
            print("Params:", model.params_)
            print("obj(init):", model._objective(model.params_, Y_mode))

    # 7) fit model iLSTM
    ilstm_models = []
    for j, idx in enumerate(ilstm_idx, start=1):
        model = iLSTM(cfg).fit(X_list[idx])
        ilstm_models.append(model)

        # naming sesuai permintaan
        # K=2  -> ilstm_model2
        # K=4  -> ilstm1_model4, ilstm2_model4
        if len(ilstm_idx) == 1:
            globals()[f"ilstm_model{K}"] = model
            print(f"\n[iLSTM model K={K}] fitted on mode {idx+1}")
        else:
            globals()[f"ilstm{j}_model{K}"] = model
            print(f"\n[iLSTM model {j} | K={K}] fitted on mode {idx+1}")

    # 8) simpan semua ke dictionary juga (lebih aman)
    all_results[K] = {
        "K": K,
        "u_low": u_low,
        "u_high": u_high,
        "u_lowhat": u_lowhat,
        "u_highhat": u_highhat,
        "omega_low": omega_low,
        "omega_high": omega_high,
        "X_list": X_list,
        "aci_idx": aci_idx,
        "ilstm_idx": ilstm_idx,
        "Y_lists": Y_lists,
        "aci_models": aci_models,
        "ilstm_models": ilstm_models,
    }

# MODEL DICT
models_dict = {}
for K in range(1, 6):
    aci_models = []
    ilstm_models = []
    for name, val in list(globals().items()):
        if f"model{K}" in name:
            if "aci" in name:
                aci_models.append((name, val))
            elif "ilstm" in name:
                ilstm_models.append((name, val))
    # sorting berdasarkan nama
    aci_models = sorted(aci_models)
    ilstm_models = sorted(ilstm_models)
    combined = [m[1] for m in aci_models + ilstm_models]
    models_dict[K] = [combined]

def decompose_interval_vmd(X, K, vmd_func, **vmd_kwargs):
    """
    Decompose interval series X (shape: T x 2) menjadi K mode dengan VMD.
    
    Parameters
    ----------
    X : np.ndarray
        Data interval, shape (T, 2), kolom = [left_bound, right_bound]
        Misal [low, high] atau [open, close].
    K : int
        Jumlah mode VMD.
    vmd_func : callable
        Fungsi VMD dengan format umum:
        u, u_hat, omega = vmd_func(signal, alpha, tau, K, DC, init, tol)
    **vmd_kwargs :
        Parameter tambahan untuk VMD selain K.

    Returns
    -------
    X_modes : list[np.ndarray]
        List panjang K.
        Setiap elemen shape (T, 2), yaitu mode ke-k untuk dua bound interval.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X harus shape (T, 2).")
    signal_left = X[:, 0]
    signal_right = X[:, 1]
    u_left, _, _ = vmd_func(signal_left, K=K, **vmd_kwargs)
    u_right, _, _ = vmd_func(signal_right, K=K, **vmd_kwargs)

    if u_left.shape[0] != K or u_right.shape[0] != K:
        raise ValueError("Output VMD tidak sesuai dengan jumlah mode K.")

    T_hist = min(len(signal_left), u_left.shape[1], u_right.shape[1])
    if K == 1:
        # Jika K=1, kita juga ingin mode residual (original - mode)
        u_left_residual = signal_left[:T_hist] - u_left[0][:T_hist]
        u_right_residual = signal_right[:T_hist] - u_right[0][:T_hist]
        X_modes = [
            np.column_stack([u_left[0][:T_hist], u_right[0][:T_hist]]),   # mode 1
            np.column_stack([u_left_residual[:T_hist], u_right_residual[:T_hist]])]  # residual
    else:
        X_modes = [
            np.column_stack([u_left[k][:T_hist], u_right[k][:T_hist]])
            for k in range(K)]
    return X_modes

def is_aci_mode(k, K):
    if K <= 2:
        return k == 0
    else:
        return k in [0, 1]

# ROLLING FORECAST
def rolling_forecast(models_list, trains_list, tests_list, K, vmd_func, n_test=None, **vmd_kwargs):
    """
    Rolling forecast sepanjang n_test steps.
    History di-update menggunakan data aktual test, bukan prediksi.

    Returns
    -------
    preds_all : list[np.ndarray]
        Tiap elemen shape (n_test, 2)
    preds_modes_all : list[list[np.ndarray]]
        preds_modes_all[i][k] shape = (n_test, 2)
    n_test : int
    """
    if n_test is None:
        n_test = min(len(x) for x in tests_list)
    else:
        n_test = int(n_test)
    preds_all = []
    preds_modes_all = []

    for models_per_k, X_train_raw, X_test_raw in zip(models_list, trains_list, tests_list):
        history_raw = np.asarray(X_train_raw, dtype=float).copy()
        X_test_raw = np.asarray(X_test_raw, dtype=float)

        # cek jumlah mode aktual
        X_modes_init = decompose_interval_vmd(
            history_raw, K=K, vmd_func=vmd_func, **vmd_kwargs
        )
        n_modes = len(X_modes_init)
        if len(models_per_k) != n_modes:
            raise ValueError(
                f"Jumlah model ({len(models_per_k)}) tidak sama dengan jumlah mode ({n_modes})"
            )
        preds_final = []
        preds_per_mode = [[] for _ in range(n_modes)]
        for t in range(n_test):
            # 1) VMD pada history saat ini
            X_modes_hist = decompose_interval_vmd(
                history_raw, K=K, vmd_func=vmd_func, **vmd_kwargs
            )
            fc_modes = []

            # 2) forecast per mode
            for k in range(n_modes):
                X_k = X_modes_hist[k]
                # ===== ACI =====
                if is_aci_mode(k, K):
                    if len(X_k) < 2:
                        raise ValueError("Data terlalu pendek untuk differencing ACI")
                    X_diff = hukuhara_diff(X_k[1:], X_k[:-1])
                    fc_diff = np.asarray(
                        models_per_k[k].forecast_1step(X_diff),
                        dtype=float
                    )
                    if fc_diff.shape != (2,):
                        raise ValueError(
                            f"Output ACI mode-{k+1} harus (2,), dapat {fc_diff.shape}"
                        )
                    fc_k = inverse_diff(fc_diff.reshape(1, 2), X_k[-1])[0]

                # ===== iLSTM =====
                else:
                    fc_k = np.asarray(
                        models_per_k[k].forecast_1step(X_k),
                        dtype=float
                    )
                    if fc_k.shape != (2,):
                        raise ValueError(
                            f"Output iLSTM mode-{k+1} harus (2,), dapat {fc_k.shape}"
                        )
                fc_modes.append(fc_k)
                preds_per_mode[k].append(fc_k)
            # 3) agregasi semua mode
            fc_final = np.sum(np.stack(fc_modes, axis=0), axis=0)
            preds_final.append(fc_final)
            # 4) update history pakai data aktual test
            history_raw = np.vstack([history_raw, X_test_raw[t]])

        preds_all.append(np.asarray(preds_final))
        preds_modes_all.append([np.asarray(p) for p in preds_per_mode])

    return preds_all, preds_modes_all, n_test

# PROSES PERAMALAN
vmd_params = {
    "alpha": 2000, 
    "tau": 0,      
    "DC": 0,       
    "init": 1,     
    "tol": 1e-7}

pred_store = {}
mode_pred_store = {}
for K in range(1, 6):
    preds_all, preds_modes_all, H = rolling_forecast(
        models_list=models_dict[K],
        trains_list=[X_train],
        tests_list=[X_test],
        K=K,
        vmd_func=VMD,
        **vmd_params
    )
    pred_store[K] = preds_all[0]
    mode_pred_store[K] = preds_modes_all[0]
