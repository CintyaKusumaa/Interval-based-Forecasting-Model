import numpy as np
from scipy.optimize import minimize

class ACI:
    """
    Autoregressive Conditional Interval model ACI(p, q) dengan opsi ACI-X (ECT).
    Parameter:
        p (int): orde AR
        q (int): orde MA
        K (2x2): matriks DK-distance. Default identitas.
    """

    def __init__(self, p=1, q=1, K=None, p1=None):
        self.p = int(p)
        self.q = int(q)
        if K is None:
            self.K = np.array([[1.0, 0.0],
                               [0.0, 1.0]], dtype=float)
        else:
            self.K = np.asarray(K, dtype=float)
        # Default: p1 = setengah dari p
        if p1 is None:
            self.p1 = p // 2
        else:
            self.p1 = p1
        self.params_ = None
        self._fitted_resid_ = None

    # ========== Interval Operations ========== #
    @staticmethod
    def interval_add(A, B):
        """Addition: [A_L + B_L, A_R + B_R]"""
        return np.array([A[0] + B[0], A[1] + B[1]], dtype=float)
    
    @staticmethod
    def interval_neg(x):
        return np.array([-x[1], -x[0]])

    @staticmethod
    def interval_scalar_mul(lam, A):
        """Scalar multiplication: λ·[A_L, A_R] = [λ·A_L, λ·A_R]"""
        lam = float(lam)
        return np.array([lam * A[0], lam * A[1]], dtype=float)

    @staticmethod
    def interval_diff(A, B):
        """
        Component-wise difference: [A_L - B_L, A_R - B_R].
        (Sering dipakai sebagai residual interval secara praktis.)
        """
        return np.array([A[0] - B[0], A[1] - B[1]], dtype=float)

    @staticmethod
    def interval_point(c):
        """Degenerate interval [c, c]."""
        c = float(c)
        return np.array([c, c], dtype=float)
    
    @staticmethod
    def interval_ect(ec):
        ec = float(ec)
        return np.array([0.5 * ec, 1.5 * ec], dtype=float)

    # ========== Core Model ========== #
    def _unpack_params(self, params):
        """
        params = [alpha0, beta0, beta_1..beta_p, gamma_1..gamma_q, delta]
        delta adalah koefisien ECT.
        """
        alpha0 = params[0]
        beta0 = params[1]
        beta = params[2:2 + self.p]
        gamma = params[2 + self.p:2 + self.p + self.q]
        delta = params[2 + self.p + self.q]
        return alpha0, beta0, beta, gamma, delta

    def _predict_conditional_mean(self, t, X, params, u_hist, ect_hist=None):
        """
        Prediksi conditional mean interval untuk X_t menggunakan informasi sampai t-1.
        ect_hist:
            array shape (T,) berisi ECT_t (skalar). Model akan memakai ect_hist[t-1].
        """
        alpha0, beta0, beta, gamma, delta = self._unpack_params(params)
        # I0 = [-0.5, 0.5] -> setara dengan [alpha0-0.5b0, alpha0+0.5b0]
        x_hat = np.array([
            alpha0 - 0.5 * beta0, 
            alpha0 + 0.5 * beta0], 
            dtype=float)
        # AR component: sum_{j=1}^p beta_j * X_{t-j}
        for j in range(1, self.p + 1):
            if t - j >= 0:
                x_hat = self.interval_add(
                    x_hat,
                    self.interval_scalar_mul(beta[j - 1], X[t - j]))
        for j in range(1, self.p + 1):
            x_lag = X[t-j]
            if j > self.p1:
                x_lag = self.interval_neg(x_lag)  # [-H, -L]
            x_hat = self.interval_add(
                x_hat,
                self.interval_scalar_mul(beta[j-1], x_lag)
            )
        # MA component: sum_{j=1}^q gamma_j * u_{t-j}
        for j in range(1, self.q + 1):
            if t - j >= 0:
                x_hat = self.interval_add(
                    x_hat,
                    self.interval_scalar_mul(gamma[j - 1], u_hist[t - j])
                )
        # ECT term (lag-1)
        if ect_hist is not None and (t - 1) >= 0:
            ect_lag = float(ect_hist[t - 1])
            ect_interval = self.interval_scalar_mul(delta, self.interval_ect(ect_lag))
            x_hat = self.interval_add(x_hat, ect_interval)
        
        return x_hat

    def _objective(self, params, X, ect_hist=None, init_residual=None):
        """
        Objective: rata-rata dk^2.
        DK distance mengikuti d=(u_R, -u_L), dk^2 = d' K d
        """
        X = np.asarray(X, dtype=float)
        T = X.shape[0]
        max_lag = max(self.p, self.q)
        BIG = 1e8
        # guard parameter
        if (not np.all(np.isfinite(params))) or (np.max(np.abs(params)) > 50):
            return BIG
        u = np.zeros_like(X)
        if init_residual is not None:
            init_residual = np.asarray(init_residual, dtype=float)
            u[:max_lag] = init_residual
        total = 0.0
        for t in range(max_lag, T):
            x_hat_t = self._predict_conditional_mean(t, X, params, u, ect_hist=ect_hist)
            # residual interval
            u_t = self.interval_diff(X[t], x_hat_t)
            u[t] = u_t
            if (not np.all(np.isfinite(u_t))) or (np.max(np.abs(u_t)) > 1e6):
                return BIG
            uL, uR = float(u_t[0]), float(u_t[1])
            dvec = np.array([uR, -uL], dtype=float)
            dk2 = float(dvec @ self.K @ dvec)
            if not np.isfinite(dk2):
                return BIG
            total += dk2
            if (not np.isfinite(total)) or (total > BIG):
                return BIG
        n_eff = T - max_lag
        return total / n_eff if n_eff > 0 else BIG

    # ========== Fitting ========== #
    def fit(self, X, ect_hist=None, initial_params=None, init_residual=None, maxiter=1000):
        """
        Fit ACI(p,q) (atau ACIX/EC-ACI jika ect_hist diberikan).
        Parameter:
            X: array (T,2) interval [L,R]
            ect_hist: array (T,) ECT_t (skalar). Model memakai lag-1: ECT_{t-1}.
            initial_params: array awal parameter (opsional)
            init_residual: residual awal untuk t < max_lag (opsional)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X harus array shape (T,2) dengan kolom [L, R].")
        T = X.shape[0]
        if ect_hist is not None:
            ect_hist = np.asarray(ect_hist, dtype=float)
            if ect_hist.shape[0] != T:
                raise ValueError("ect_hist harus punya panjang yang sama dengan X (T).")
        n_params = 2 + self.p + self.q + 1  # +1 delta (ECT)

        if initial_params is None:
            initial_params = np.zeros(n_params, dtype=float)
            # alpha0 ~ rata-rata midpoint
            mid = X.mean(axis=1).mean()
            initial_params[0] = mid
            # beta0,beta,gamma,delta default 0

        # bounds sederhana untuk stabilitas numerik
        bounds = []
        a0 = float(initial_params[0])
        bounds.append((a0 - 10 * abs(a0 + 1e-6), 
                       a0 + 10 * abs(a0 + 1e-6)))  # alpha0
        bounds.append((-10.0, 10.0))  # beta0
        for _ in range(self.p):
            bounds.append((-5.0, 5.0))  # beta_j
        for _ in range(self.q):
            bounds.append((-5.0, 5.0))  # gamma_j
        bounds.append((-5.0, 5.0))  # delta

        res = minimize(
            fun=self._objective,
            x0=np.asarray(initial_params, dtype=float),
            args=(X, ect_hist, init_residual),
            method="L-BFGS-B",
            bounds=bounds,
            options={"disp": False, "maxiter": int(maxiter)}
        )
        if not res.success:
            # tetap simpan hasil terbaik yang ada
            print(f"[ACI] Warning: Optimization did not fully converge. Message: {res.message}")
        self.params_ = res.x
        self._fitted_resid_ = self._compute_residuals(X, self.params_, ect_hist=ect_hist, init_residual=init_residual)
        return self

    def _compute_residuals(self, X, params, ect_hist=None, init_residual=None):
        """Hitung residual u_t untuk semua t."""
        X = np.asarray(X, dtype=float)
        T = X.shape[0]
        max_lag = max(self.p, self.q)
        u = np.zeros_like(X)
        if init_residual is not None:
            init_residual = np.asarray(init_residual, dtype=float)
            u[:max_lag] = init_residual
        for t in range(max_lag, T):
            x_hat_t = self._predict_conditional_mean(t, X, params, u, ect_hist=ect_hist)
            u[t] = self.interval_diff(X[t], x_hat_t)
        return u

    def predict_in_sample(self, X, ect_hist=None):
        """
        In-sample fitted values.
        """
        if self.params_ is None:
            raise RuntimeError("Model belum di-fit.")
        X = np.asarray(X, dtype=float)
        T = X.shape[0]
        max_lag = max(self.p, self.q)
        if ect_hist is not None:
            ect_hist = np.asarray(ect_hist, dtype=float)
            if ect_hist.shape[0] != T:
                raise ValueError("ect_hist harus punya panjang yang sama dengan X (T).")
        X_hat = np.zeros_like(X)
        u = np.zeros_like(X)
        for t in range(max_lag, T):
            X_hat[t] = self._predict_conditional_mean(t, X, self.params_, u, ect_hist=ect_hist)
            u[t] = self.interval_diff(X[t], X_hat[t])
        return X_hat
    
    def forecast_1step(self, X, ect_last=None, ect_hist=None, init_residual=None):
        """
        Forecast 1-step ahead untuk X_T berdasarkan history sampai X_{T-1}.

        Parameters
        ----------
        X : array (T,2)
            History series yang dipakai model.
        ect_last : float, optional
            Nilai ECT terakhir, yaitu ECT_{T-1}, untuk 1-step ahead.
        ect_hist : array (T,), optional
            History ECT penuh yang aligned dengan X.
            Kalau diberikan, residual historis dihitung konsisten dengan ECT.
        """
        if self.params_ is None:
            raise RuntimeError("Model belum di-fit.")

        X = np.asarray(X, dtype=float)
        T = X.shape[0]

        if T < max(self.p, self.q) + 1:
            raise ValueError("History terlalu pendek untuk forecast ACI(p,q).")

        # residual sejarah HARUS konsisten dengan ect_hist bila model pakai ECT
        u = self._compute_residuals(
            X,
            self.params_,
            ect_hist=ect_hist,
            init_residual=init_residual
        )

        ect_hist_full = None
        if ect_last is not None:
            ect_hist_full = np.zeros(T + 1, dtype=float)
            if ect_hist is not None:
                ect_hist = np.asarray(ect_hist, dtype=float).reshape(-1)
                if len(ect_hist) != T:
                    raise ValueError("ect_hist harus punya panjang sama dengan X.")
                ect_hist_full[:T] = ect_hist
            ect_hist_full[T - 1] = float(ect_last)

        x_hat = self._predict_conditional_mean(
            T,
            np.vstack([X, np.zeros((1, 2))]),
            self.params_,
            np.vstack([u, np.zeros((1, 2))]),
            ect_hist=ect_hist_full
        )
        return x_hat
    
    def forecast(self, X, steps=1, ect_future=None):
        assert self.params_ is not None, "Model not fitted"

        X = np.asarray(X, dtype=float)
        T = X.shape[0]
        steps = int(steps)

        u = self._compute_residuals(X, self.params_)

        X_extended = np.vstack([X, np.zeros((steps, 2))])
        u_extended = np.vstack([u, np.zeros((steps, 2))])

        if ect_future is not None:
            ect_future = np.asarray(ect_future, dtype=float)
            if ect_future.shape[0] != steps:
                raise ValueError("ect_future must have length = steps")

        forecasts = np.zeros((steps, 2), dtype=float)

        for h in range(steps):
            t = T + h

            if ect_future is None:
                x_hat = self._predict_conditional_mean(
                    t, X_extended, self.params_, u_extended
                )
            else:
                ect_hist = np.concatenate([np.zeros(T), ect_future])
                x_hat = self._predict_conditional_mean(
                    t, X_extended, self.params_, u_extended, ect_hist=ect_hist
                )

            forecasts[h] = x_hat
            X_extended[t] = x_hat
            u_extended[t] = np.array([0.0, 0.0])

        return forecasts

    @property
    def residuals_(self):
        """Residual in-sample setelah fit (shape (T,2))."""
        return self._fitted_resid_