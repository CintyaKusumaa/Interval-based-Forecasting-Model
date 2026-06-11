# PREPARE DATA: RETURN SERIES
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# return dalam persen
df["log_return_low"]   = np.log(df["Low"]).diff() * 100
df["log_return_high"]  = np.log(df["High"]).diff() * 100
df["log_return_close"] = np.log(df["Close"]).diff() * 100

# level price series
low_series   = pd.Series(df["Low"].values,   index=df["Date"], name="Low")
high_series  = pd.Series(df["High"].values,  index=df["Date"], name="High")
close_series = pd.Series(df["Close"].values, index=df["Date"], name="Close")

# return series
r_low_series = pd.Series(df["log_return_low"].values, index=df["Date"], name="r_low").dropna()
r_high_series = pd.Series(df["log_return_high"].values, index=df["Date"], name="r_high").dropna()
r_close_series = pd.Series(df["log_return_close"].values, index=df["Date"], name="r_close").dropna()

# SPLIT TRAIN-TEST
def split_series(series, train_ratio=0.8):
    split_idx = int(train_ratio * len(series))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test
rlow_train, rlow_test     = split_series(r_low_series, train_ratio=0.8)
rhigh_train, rhigh_test   = split_series(r_high_series, train_ratio=0.8)
rclose_train, rclose_test = split_series(r_close_series, train_ratio=0.8)

# CHECK ARMA NEEDED OR NOT
from statsmodels.stats.diagnostic import acorr_ljungbox
def check_arma_needed(train_series, max_lag=10, alpha=0.05):
    ljung = acorr_ljungbox(train_series, lags=range(1, max_lag + 1), return_df=True)
    arma_status = (ljung["lb_pvalue"] < alpha).any()
    return arma_status, ljung
arma_status_low, ljung_low = check_arma_needed(rlow_train)
arma_status_high, ljung_high = check_arma_needed(rhigh_train)
arma_status_close, ljung_close = check_arma_needed(rclose_train)

# SELECT BEST ARMA ORDER
from statsmodels.tsa.arima.model import ARIMA
def find_best_arma_order(train_series, p_range=range(1,4), q_range=range(1,4)):
    best_aic = float("inf")
    best_order = None
    best_model = None
    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(train_series, order=(p, 0, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, q)
                    best_model = model
            except:
                continue
    return best_order, best_model, best_aic

# SELECT BEST GARCH ORDER
from arch import arch_model
def find_best_garch_order(data_input, mean_type="Constant", candidates=[(1,1), (1,2), (2,1), (2,2)]):
    results = []
    for p, q in candidates:
        try:
            model = arch_model(
                data_input,
                mean=mean_type,
                vol="GARCH",
                p=p,
                q=q,
                dist="normal"
            )
            res = model.fit(disp="off")
            alpha_sum = res.params[[k for k in res.params.index if "alpha" in k]].sum()
            beta_sum  = res.params[[k for k in res.params.index if "beta" in k]].sum()
            results.append({
                "p": p,
                "q": q,
                "AIC": res.aic,
                "BIC": res.bic,
                "alpha": alpha_sum,
                "beta": beta_sum,
                "alpha+beta": alpha_sum + beta_sum
            })
        except:
            continue
    garch_df = pd.DataFrame(results).sort_values(["BIC", "AIC"]).reset_index(drop=True)
    return garch_df


# MAIN FUNCTION: ROLLING ARMA-GARCH FORECAST
def rolling_armagarch_pipeline(return_series, level_series, name="series", train_ratio=0.8):
    # split
    train_ret, test_ret = split_series(return_series, train_ratio=train_ratio)
    # previous price for inverse transform later
    prev_price = level_series.shift(1).loc[test_ret.index]
    
    # check ARMA needed
    arma_status, ljung_df = check_arma_needed(train_ret)
    if arma_status:
        best_order, best_model, best_aic = find_best_arma_order(train_ret)
        resid_train = best_model.resid
        data_input = resid_train
        mean_type = "Zero"
    else:
        best_order, best_model, best_aic = None, None, None
        data_input = train_ret
        mean_type = "Constant"

    # select best GARCH
    garch_df = find_best_garch_order(data_input, mean_type=mean_type)
    p_gc = int(garch_df.loc[0, "p"])
    q_gc = int(garch_df.loc[0, "q"])

    # fit in-sample model
    if arma_status:
        arma_init = ARIMA(train_ret, order=(best_order[0], 0, best_order[1])).fit()
        resid_init = arma_init.resid
        garch_init = arch_model(
            resid_init,
            mean="Zero",
            vol="GARCH",
            p=p_gc,
            q=q_gc,
            dist="normal"
        )
        res_garch_init = garch_init.fit(disp="off")
        insample_vol = pd.Series(
            res_garch_init.conditional_volatility,
            index=train_ret.index,
            name=f"insample_vol_{name}"
        )
    else:
        garch_init = arch_model(
            train_ret,
            mean="Constant",
            vol="GARCH",
            p=p_gc,
            q=q_gc,
            dist="normal"
        )
        res_garch_init = garch_init.fit(disp="off")
        insample_vol = pd.Series(
            res_garch_init.conditional_volatility,
            index=train_ret.index,
            name=f"insample_vol_{name}"
        )

    # rolling forecast
    history_r = train_ret.copy()
    mean_fc_list = []
    vol_fc_list = []
    var_fc_list = []

    for t in range(len(test_ret)):
        try:
            if arma_status:
                arma_model = ARIMA(
                    history_r,
                    order=(best_order[0], 0, best_order[1])
                ).fit()
                # mean forecast dari ARMA
                arma_fc = arma_model.get_forecast(steps=1)
                mean_fc = float(arma_fc.predicted_mean.iloc[0])
                resid = arma_model.resid
                garch_model = arch_model(
                    resid,
                    mean="Zero",
                    vol="GARCH",
                    p=p_gc,
                    q=q_gc,
                    dist="normal"
                )
                garch_res = garch_model.fit(disp="off")
                fc = garch_res.forecast(horizon=1, reindex=False)
                var_fc = float(fc.variance.iloc[-1, 0])
                vol_fc = np.sqrt(var_fc)
            else:
                garch_model = arch_model(
                    history_r,
                    mean="Constant",
                    vol="GARCH",
                    p=p_gc,
                    q=q_gc,
                    dist="normal"
                )
                garch_res = garch_model.fit(disp="off")
                fc = garch_res.forecast(horizon=1, reindex=False)
                
                # mean forecast dari GARCH jika tanpa ARMA
                mean_fc = float(fc.mean.iloc[-1, 0])
                var_fc = float(fc.variance.iloc[-1, 0])
                vol_fc = np.sqrt(var_fc)
            mean_fc_list.append(mean_fc)
            vol_fc_list.append(vol_fc)
            var_fc_list.append(var_fc)
            
        except Exception as e:
            print(f"[{name}] Step {t} error: {e}")
            mean_fc_list.append(np.nan)
            vol_fc_list.append(np.nan)
            var_fc_list.append(np.nan)
        # expanding window update
        new_obs = test_ret.iloc[t:t+1]
        history_r = pd.concat([history_r, new_obs])
    
    # output dataframe
    forecast_df = pd.DataFrame({
        "actual_return": test_ret,
        "mean_forecast_return": mean_fc_list,
        "var_forecast": var_fc_list,
        "vol_forecast": vol_fc_list,
        "prev_price": prev_price
    }, index=test_ret.index)
    return {
        "name": name,
        "train_return": train_ret,
        "test_return": test_ret,
        "arma_status": arma_status,
        "ljungbox": ljung_df,
        "best_order": best_order,
        "best_aic": best_aic,
        "garch_df": garch_df,
        "p_gc": p_gc,
        "q_gc": q_gc,
        "insample_vol": insample_vol,
        "forecast_df": forecast_df
    }

res_low = rolling_armagarch_pipeline(return_series=r_low_series,
    level_series=low_series, name="low",
    train_ratio=0.8)
res_high = rolling_armagarch_pipeline(return_series=r_high_series,
    level_series=high_series, name="high",
    train_ratio=0.8)
res_close = rolling_armagarch_pipeline(return_series=r_close_series,
    level_series=close_series, name="close",
    train_ratio=0.8)

# INVERSE RETURN FORECAST TO ORIGINAL PRICE SCALE
def inverse_return_forecast_to_price(result_dict, level_series):
    """
    level_series: original price series (Low / High / Open / Close)
    result_dict : output dari rolling_armagarch_pipeline()
    """
    forecast_df = result_dict["forecast_df"].copy()

    # actual price pada periode test
    actual_price = level_series.loc[forecast_df.index]

    # inverse transform:
    # r_t = 100 * (log(P_t) - log(P_{t-1}))
    # => P_t = P_{t-1} * exp(r_t / 100)
    pred_price = forecast_df["prev_price"] * np.exp(forecast_df["mean_forecast_return"] / 100)

    forecast_df["actual_price"] = actual_price
    forecast_df["forecast_price"] = pred_price
    forecast_df["naive_price"] = forecast_df["prev_price"]  # random walk benchmark
    result_dict["forecast_df"] = forecast_df
    return result_dict

res_low   = inverse_return_forecast_to_price(res_low, low_series)
res_high  = inverse_return_forecast_to_price(res_high, high_series)
res_close = inverse_return_forecast_to_price(res_close, close_series)

# POINT FORECAST METRICS
def evaluate_point_forecast(result_dict):
    df_fc = result_dict["forecast_df"].dropna().copy()
    y_true = df_fc["actual_price"].values
    y_pred = df_fc["forecast_price"].values
    y_naive = df_fc["naive_price"].values
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Theil U vs random walk
    ui = np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_naive) ** 2))

    # ARVI vs mean benchmark
    y_mean = np.mean(y_true)
    arvi = np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_mean) ** 2)

    # Directional accuracy
    # bandingkan arah perubahan prediksi vs aktual terhadap harga sebelumnya
    actual_dir = np.sign(y_true - y_naive)
    pred_dir   = np.sign(y_pred - y_naive)
    da = np.mean(actual_dir == pred_dir) * 100

    return pd.DataFrame([{
        "Series": result_dict["name"],
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "UI": ui,
        "ARVI": arvi,
        "DA (%)": da
    }])

eval_point_low   = evaluate_point_forecast(res_low)
eval_point_high  = evaluate_point_forecast(res_high)
eval_point_close = evaluate_point_forecast(res_close)
eval_point_all = pd.concat(
    [eval_point_low, eval_point_high, eval_point_close],
    ignore_index=True)
eval_point_all