"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

# import additional ...
import math
from pathlib import Path

import gpytorch
import gpytorch.constraints
import matplotlib.pyplot as plt
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions import Normal
from torch.optim import Adam


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
MIN_INIT_DIST = 0.1  # minimum distance from initial point
MIN_INIT_DIST_EPS = 1e-6  # minimum distance from initial point (epsilon)
MIN_AF_VALUE = torch.finfo(torch.float32).min  # minimum value of AF

PLOT_ALL = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_initializations = 0


def log1mexp(Z: torch.Tensor) -> torch.Tensor:
    """Computes log(1 - exp(Z)) for Z > 0 in a numerically stable manner.

    References
    ----------
        Mächler, M. (2015). Accurately Computing log(1 - exp(- |a|)) Assessed
        by the Rmpfr package. 10.13140/RG.2.2.11834.70084.
    """
    return torch.where(
        Z >= -math.log(2),
        torch.log(-torch.expm1(Z)),
        torch.log1p(-torch.exp(Z)),
    )


def h(Z: torch.Tensor) -> torch.Tensor:
    """Computes phi(Z) + Z * Phi(Z)."""
    dist = Normal(torch.zeros_like(Z), torch.ones_like(Z))
    return torch.exp(dist.log_prob(Z)) + Z * dist.cdf(Z)


def log_h(Z: torch.Tensor) -> torch.Tensor:
    """Computes log(phi(Z) + Z * Phi(Z)) in a numerically stable manner.

    References
    ----------
        Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2024).
        Unexpected Improvements to Expected Improvement for Bayesian Optimization.
        arXiv preprint arXiv:2310.20708.
    """
    U1 = -(torch.finfo(Z.dtype).resolution ** -0.5)
    U2 = -1.0

    Z1 = torch.clamp(Z, None, U1)
    Z2 = torch.clamp(Z, U1, U2)
    Z3 = torch.clamp(Z, U2, None)

    dist = Normal(torch.zeros_like(Z), torch.ones_like(Z))

    return torch.where(
        Z <= U1,
        dist.log_prob(Z1) - 2 * torch.log(torch.abs(Z1)),
        torch.where(
            Z <= U2,
            dist.log_prob(Z2) + log1mexp(
                torch.log(torch.special.erfcx(-(2**-0.5) * Z2) * torch.abs(Z2))
                + math.log(math.pi / 2) / 2
            ),
            torch.log(h(Z3)),
        ),
    )


def ei_f(out_f: gpytorch.distributions.MultivariateNormal, f_best: torch.Tensor, xi: float) -> torch.Tensor:
    """Computes EI(x) = sigma(x) * h((mu(x) - f_best - xi) / sigma(x))."""
    Z = (out_f.mean - f_best - xi) / out_f.stddev
    return out_f.stddev * h(Z)


def log_ei_f(out_f: gpytorch.distributions.MultivariateNormal, f_best: torch.Tensor, xi: float) -> torch.Tensor:
    """Computes log(EI(x)) = log(sigma(x)) + log(h((mu(x) - f_best - xi) / sigma(x)))."""
    Z = (out_f.mean - f_best - xi) / out_f.stddev
    return torch.log(out_f.stddev) + log_h(Z)


def pr_v(out_v: gpytorch.distributions.MultivariateNormal) -> torch.Tensor:
    """Computes Pr(v(x) < kappa) = Phi((kappa - mu(x)) / sigma(x))."""
    dist = Normal(torch.zeros_like(out_v.mean), torch.ones_like(out_v.mean))
    Z = (SAFETY_THRESHOLD - out_v.mean) / out_v.stddev
    return dist.cdf(Z)


def log_pr_v(out_v: gpytorch.distributions.MultivariateNormal) -> torch.Tensor:
    """Computes log(Pr(v(x) < kappa)) = log(Phi((kappa - mu(x)) / sigma(x)))."""
    Z = (SAFETY_THRESHOLD - out_v.mean) / out_v.stddev
    return torch.special.log_ndtr(Z)


def af(
    out_f: gpytorch.distributions.MultivariateNormal,
    out_v: gpytorch.distributions.MultivariateNormal,
    f_best: torch.Tensor,
    xi: float,
    delta: float,
) -> torch.Tensor:
    """Computes AF(x) = EI(x) * Pr(v(x) < kappa)."""
    log_ei_f_value = log_ei_f(out_f, f_best, xi)
    log_pr_v_value = log_pr_v(out_v)
    pr_v_value = pr_v(out_v)
    return torch.where(
        pr_v_value >= 1 - delta,
        log_pr_v_value + log_ei_f_value,
        MIN_AF_VALUE,
    )


class ExactGPModelF(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelF, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()

        rbf_kernel = gpytorch.kernels.RBFKernel()
        rbf_kernel.initialize(lengthscale=1.0)

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel=rbf_kernel)
        self.covar_module.initialize(outputscale=1.0)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelV(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelV, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.initialize(constant=4.0)

        linear_kernel = gpytorch.kernels.LinearKernel()

        matern_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        matern_kernel.initialize(lengthscale=0.5)

        self.covar_module = linear_kernel + matern_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# DONE: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
class BOAlgorithm():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # DONE: Define all relevant class members for your BO algorithm here.
        train_x = torch.empty((0, 1), dtype=torch.float32, device=device)
        train_y = torch.empty((0,), dtype=torch.float32, device=device)

        likelihood_f = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8)).to(device)
        likelihood_f.initialize(noise=0.15 ** 2)

        likelihood_v = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8)).to(device)
        likelihood_v.initialize(noise=0.0001 ** 2)

        self.gp_f = ExactGPModelF(train_x, train_y, likelihood_f).to(device)
        self.gp_v = ExactGPModelV(train_x, train_y, likelihood_v).to(device)

        self.optimizer_f = Adam(self.gp_f.parameters(), lr=0.3)
        self.optimizer_v = Adam(self.gp_v.parameters(), lr=0.3)

        self.mll_f = ExactMarginalLogLikelihood(self.gp_f.likelihood, self.gp_f)
        self.mll_v = ExactMarginalLogLikelihood(self.gp_v.likelihood, self.gp_v)

        self.xi = 0.1
        self.delta = 1e-11
        self.training_iter = 0

        self.f_best = None
        self.ind_best = None
        self.x_best = None

        global num_initializations
        self.dataset_name = f'private_{num_initializations}'
        num_initializations += 1

    def recommend_next(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # DONE: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        x_opts = []
        af_opts = []

        for _ in range(15):
            while True:
                x_opt = self.optimize_acquisition_function()
                af_opt = self.acquisition_function(np.asarray([[x_opt]])).item()
                if af_opt != MIN_AF_VALUE:
                    break
            x_opts.append(x_opt)
            af_opts.append(af_opt)

        x_opts = np.asarray(x_opts)
        af_opts = np.asarray(af_opts)

        ind_opt = af_opts.argmax()
        return x_opts[ind_opt].item()

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)

        # DONE: Implement the acquisition function you want to optimize.
        x = torch.as_tensor(x, dtype=torch.float32, device=device)

        self.gp_f.eval()
        self.gp_f.likelihood.eval()
        self.gp_v.eval()
        self.gp_v.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out_f = self.gp_f(x)
            out_v = self.gp_v(x)

        af_value = af(out_f, out_v, self.f_best, self.xi, self.delta)
        return af_value.unsqueeze(-1).detach().cpu().numpy()

    def add_observation(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # DONE: Add the observed data {x, f, v} to your model.
        x = np.atleast_1d(x)
        f = np.atleast_1d(f)
        v = np.atleast_1d(v)

        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        f = torch.as_tensor(f, dtype=torch.float32, device=device)
        v = torch.as_tensor(v, dtype=torch.float32, device=device)

        inputs_f = torch.cat((self.gp_f.train_inputs[0], x[..., None]))
        inputs_v = torch.cat((self.gp_v.train_inputs[0], x[..., None]))

        targets_f = torch.cat((self.gp_f.train_targets, f))
        targets_v = torch.cat((self.gp_v.train_targets, v))

        self.gp_f.set_train_data(inputs_f, targets_f, strict=False)
        self.gp_v.set_train_data(inputs_v, targets_v, strict=False)

        if self.training_iter > 0:
            self.gp_f.train()
            self.gp_f.likelihood.train()
            self.gp_v.train()
            self.gp_v.likelihood.train()

            for i in range(self.training_iter):
                self.optimizer_f.zero_grad()
                self.optimizer_v.zero_grad()

                out_f = self.gp_f(self.gp_f.train_inputs[0])
                out_v = self.gp_v(self.gp_v.train_inputs[0])

                loss_f = -self.mll_f(out_f, self.gp_f.train_targets)
                loss_v = -self.mll_v(out_v, self.gp_v.train_targets)

                loss_f.backward()
                loss_v.backward()

                print(
                    f'Iteration {i + 1}/{self.training_iter} - '
                    f'Loss f: {loss_f.item():.3f} - Loss v: {loss_v.item():.3f} - '
                    f'Mean v: {self.gp_v.mean_module.constant.item():.3f} - '
                    f'Variance v: {self.gp_v.covar_module.kernels[0].variance.item():.3f} - '
                    f'Outputscale f: {self.gp_f.covar_module.outputscale.item():.3f} - '
                    f'Lengthscale f: {self.gp_f.covar_module.base_kernel.lengthscale.item():.3f} - Lengthscale v: {self.gp_v.covar_module.kernels[1].lengthscale.item():.3f} - '
                    f'Noise f: {self.gp_f.likelihood.noise.item():.3f} - Noise v: {self.gp_v.likelihood.noise.item():.3f}'
                )

                self.optimizer_f.step()
                self.optimizer_v.step()

        self.gp_f.eval()
        self.gp_f.likelihood.eval()
        self.gp_v.eval()
        self.gp_v.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out_f = self.gp_f(self.gp_f.train_inputs[0])
            out_v = self.gp_v(self.gp_v.train_inputs[0])

        x_init = self.gp_f.train_inputs[0][0]
        safe_idx = out_v.mean + 3*out_v.stddev < SAFETY_THRESHOLD
        far_idx = torch.abs(self.gp_f.train_inputs[0][safe_idx] - x_init) >= MIN_INIT_DIST + MIN_INIT_DIST_EPS
        safe_f_values = out_f.mean[safe_idx][far_idx.squeeze(-1)]
        if safe_f_values.shape[0] == 0:
            safe_f_values = out_f.mean[safe_idx]
        self.f_best = safe_f_values.max()
        idx = out_f.mean == self.f_best
        train_f_best = self.gp_f.train_targets[idx].max()
        self.ind_best = torch.where(self.gp_f.train_targets == train_f_best)[0][0].item()
        self.x_best = self.gp_f.train_inputs[0][self.ind_best]

        if PLOT_ALL:
            self.plot()

    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # DONE: Return your predicted safe optimum of f.
        return self.x_best.item()

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        self.gp_f.eval()
        self.gp_f.likelihood.eval()
        self.gp_v.eval()
        self.gp_v.likelihood.eval()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
        x = np.linspace(DOMAIN[:, 0].item(), DOMAIN[:, 1].item(), 100)

        if self.dataset_name == 'toy':
            true_f_values = np.array([f(x_i) for x_i in x])
            true_v_values = np.array([v(x_i) for x_i in x])

            # Plot ground-truth
            ax1.plot(x, true_f_values, ls='--', color='C0', label=r'Ground truth ($f$)')
            ax1.plot(x, true_v_values, ls='--', color='C1', label=r'Ground truth ($v$)')

            # Plot the maximum of f in the domain where v < threshold
            safe_idx = true_v_values < SAFETY_THRESHOLD
            safe_x = x[safe_idx]
            safe_f_values = true_f_values[safe_idx]
            safe_v_values = true_v_values[safe_idx]
            true_ind_f_opt = safe_f_values.argmax()
            ax1.scatter(safe_x[true_ind_f_opt], safe_f_values[true_ind_f_opt], marker='*', s=10 ** 2, color='green', label=r'Safe ground-truth optimum ($f$)')
            ax1.scatter(safe_x[true_ind_f_opt], safe_v_values[true_ind_f_opt], marker='*', s=10 ** 2, color='green', label=r'Safe ground-truth optimum ($v$)')

        # Plot gp prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_f = self.gp_f(torch.as_tensor(x[..., np.newaxis], dtype=torch.float32, device=device))
            pred_v = self.gp_v(torch.as_tensor(x[..., np.newaxis], dtype=torch.float32, device=device))
        mean_f = pred_f.mean.detach().cpu().numpy()
        mean_v = pred_v.mean.detach().cpu().numpy()
        error_f = 2 * pred_f.stddev.detach().cpu().numpy()
        error_v = 2 * pred_v.stddev.detach().cpu().numpy()
        ax1.fill_between(x, mean_f - error_f, mean_f + error_f, lw=0, alpha=0.2, color='C0')
        ax1.fill_between(x, mean_v - error_v, mean_v + error_v, lw=0, alpha=0.2, color='C1')
        ax1.plot(x, mean_f, lw=2, color='C0', label=r'Prediction ($f$)')
        ax1.plot(x, mean_v, lw=2, color='C1', label=r'Prediction ($v$)')

        # Plot optimal solution
        inputs_f = self.gp_f.train_inputs[0].squeeze(-1).detach().cpu().numpy()
        inputs_v = self.gp_v.train_inputs[0].squeeze(-1).detach().cpu().numpy()
        targets_f = self.gp_f.train_targets.detach().cpu().numpy()
        targets_v = self.gp_v.train_targets.detach().cpu().numpy()
        x_opt = self.get_optimal_solution()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_opt = self.gp_f(torch.as_tensor([[x_opt]], dtype=torch.float32, device=device)).mean.item()
            v_opt = self.gp_v(torch.as_tensor([[x_opt]], dtype=torch.float32, device=device)).mean.item()
        ax1.scatter(x_opt, f_opt, marker='*', s=10 ** 2, color='C0', label=r'Optimal solution ($f$)')
        ax1.scatter(x_opt, v_opt, marker='*', s=10 ** 2, color='C1', label=r'Optimal solution ($v$)')

        # Plot start points
        inputs_f = self.gp_f.train_inputs[0].squeeze(-1).detach().cpu().numpy()
        inputs_v = self.gp_v.train_inputs[0].squeeze(-1).detach().cpu().numpy()
        targets_f = self.gp_f.train_targets.detach().cpu().numpy()
        targets_v = self.gp_v.train_targets.detach().cpu().numpy()
        ax1.scatter(inputs_f[0], targets_f[0], marker='s', s=5 ** 2, color='C0', label=r'Initial safe point ($f$)')
        ax1.scatter(inputs_v[0], targets_v[0], marker='s', s=5 ** 2, color='C1', label=r'Initial safe point ($v$)')

        # Plot data
        inputs_f = self.gp_f.train_inputs[0].squeeze(-1).detach().cpu().numpy()
        inputs_v = self.gp_v.train_inputs[0].squeeze(-1).detach().cpu().numpy()
        targets_f = self.gp_f.train_targets.detach().cpu().numpy()
        targets_v = self.gp_v.train_targets.detach().cpu().numpy()
        inputs_f = inputs_f[1:]
        inputs_v = inputs_v[1:]
        targets_f = targets_f[1:]
        targets_v = targets_v[1:]
        if len(inputs_f) > 0:
            alpha_f = np.linspace(0.25, 1, len(inputs_f))
            color_f = ['C0' if b else 'red' for b in (targets_v < SAFETY_THRESHOLD)]
            ax1.scatter(inputs_f, targets_f, marker='.', s=15 ** 2, color=color_f, alpha=alpha_f)
        if len(inputs_v) > 0:
            alpha_v = np.linspace(0.25, 1, len(inputs_v))
            color_v = ['C1' if b else 'red' for b in (targets_v < SAFETY_THRESHOLD)]
            ax1.scatter(inputs_v, targets_v, marker='.', s=15 ** 2, color=color_v, alpha=alpha_v)

        # Plot threshold
        ax1.plot(x, [SAFETY_THRESHOLD] * len(x), ls='--', color='red', label=r'Safety threshold ($\kappa$)')

        # Plot recommendation
        if plot_recommendation:
            x_rec = self.recommend_next()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                f_rec = self.gp_f(torch.as_tensor([[x_rec]], dtype=torch.float32, device=device))
                v_rec = self.gp_v(torch.as_tensor([[x_rec]], dtype=torch.float32, device=device))
            pr_v_rec = pr_v(v_rec).item()
            min_delta_rec = 1 - pr_v_rec
            ax1.scatter(x_rec, f_rec.mean.detach().cpu().numpy(), marker='>', s=10 ** 2, color='C0', label=r'Recommendation ($f$)')
            ax1.scatter(x_rec, v_rec.mean.detach().cpu().numpy(), marker='>', s=10 ** 2, color='C1', label=r'Recommendation ($v$)')
            ax1.text(0.5, -5.0, rf'Min $\delta_\mathrm{{rec}}$ = {min_delta_rec}')

        # Plot acquisition function
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out_f = self.gp_f(torch.as_tensor(x[..., np.newaxis], dtype=torch.float32, device=device))
            out_v = self.gp_v(torch.as_tensor(x[..., np.newaxis], dtype=torch.float32, device=device))
        log_ei_f_values = log_ei_f(out_f, self.f_best, self.xi).detach().cpu().numpy()
        log_pr_v_values = log_pr_v(out_v).detach().cpu().numpy()
        af_values = self.acquisition_function(x[..., np.newaxis]).squeeze(-1)
        af_values[af_values == MIN_AF_VALUE] = np.nan
        ax2.plot(x, log_ei_f_values, ls='--', color='C0', label=r'$\log\operatorname{EI}(f(x))$')
        ax2.plot(x, log_pr_v_values, ls='--', color='C1', label=r'$\log\operatorname{Pr}(v(x) < \kappa)$')
        ax2.plot(x, af_values, lw=2, color='k', label=r'$\operatorname{AF}(x)$')
        if plot_recommendation:
            af_rec = self.acquisition_function(np.asarray([[x_rec]])).item()
            ax2.scatter(x_rec, af_rec, marker='>', s=8 ** 2, color='k', label=r'Recommendation (AF)')

        # Show unsafe evaluations
        unsafe_evals = torch.sum(self.gp_v.train_targets >= SAFETY_THRESHOLD).item()
        ax1.text(0.5, -5.5, f'Unsafe evaluations: {unsafe_evals}')

        j = self.gp_f.train_inputs[0].shape[0] - 1
        ax1.set_title(f'Point {j} ({self.dataset_name})')

        ax1.set_xlim(x[0], x[-1])
        ax1.set_ylim(-6, 6)
        ax1.set_ylabel(r"Objective $f$")
        ax1.set_xlabel(r"Input $x$")
        box1 = ax1.get_position()
        ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax2.set_xlim(x[0], x[-1])
        box2 = ax2.get_position()
        ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        path = Path('.' if self.dataset_name == 'toy' else f'/results')
        path = path / f'plots/{self.dataset_name}/{j}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)

        plt.close(fig)


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return math.e ** (-(x - 5.5) ** 2) + 3.5


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BOAlgorithm()
    agent.dataset_name = 'toy'

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_observation(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for _ in range(20):
        # Get next recommendation
        x = agent.recommend_next()

        # Check for valid shape
        # assert x.shape == (1, DOMAIN.shape[0]), \
        #     f"The function recommend_next must return a numpy array of " \
        #     f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal(scale=0.15)
        cost_val = v(x) + np.random.normal(scale=0.0001)
        agent.add_observation(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_optimal_solution()
    assert check_in_domain(solution), \
        f'The function get_optimal_solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    # Compute unsafe evaluations
    unsafe_evals = torch.sum(agent.gp_v.train_targets >= SAFETY_THRESHOLD).item()

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals {unsafe_evals}\n')


if __name__ == "__main__":
    main()
