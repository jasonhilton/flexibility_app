import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.regression.linear_model import OLS


class FlexibilityApp:
    def __init__(self, true_complexity=6, N=1000):
        self.Bx = None
        self.N = N
        self.X = np.linspace(-1, 1, N)
        self.prng = np.random.default_rng()
        self.true_complexity = true_complexity

        self.generate_data()

        # Widgets
        self.N_obs = widgets.IntSlider(value=50, min=10, max=200, step=10,
                                       layout=widgets.Layout(width="600px"),
                                       style={'description_width': '200px'},
                                       description="N_observed:")
        self.complexity = widgets.IntSlider(value=1, min=0, max=20, step=1,
                                            layout=widgets.Layout(width="600px"),
                                            style={'description_width': '200px'},
                                            description="Model Complexity:")
        self.noise_sd = widgets.FloatSlider(value=0.2, min=0.1, max=1, step=0.1,
                                            layout=widgets.Layout(width="600px"),
                                            style={'description_width': '200px'},
                                            description="Noise Standard Deviation:")
        self.button = widgets.Button(description="Regenerate Observations",
                                     layout=widgets.Layout(width="250px", height="60px"))
        self.true_button = widgets.Button(description="Regenerate True Function",
                                          layout=widgets.Layout(width="250px", height="60px"))
        self.out = widgets.Output()
        self.info = widgets.HTML()

        # Wiring
        self.button.on_click(self.generate_observations)
        self.true_button.on_click(self.regenerate_everything)
        self.complexity.observe(self.update_plot, names='value')
        self.noise_sd.observe(self.generate_observations, names='value')
        self.N_obs.observe(self.generate_observations, names='value')

    def true_f(self, x):
        y = self.Bx.transform(x) @ self.beta
        return y

    def regenerate_everything(self, _=None):
        self.generate_data()
        self.generate_observations()

    def generate_data(self):
        self.Bx = BSplines(self.X, self.true_complexity, degree=3)
        self.beta = self.prng.normal(size=self.true_complexity - 1)
        self.f_x = self.true_f(self.X)

    def generate_observations(self, _=None):
        self.obs_int = self.prng.choice(np.arange(self.N), self.N_obs.value, replace=False)
        self.X_obs = self.X[self.obs_int]
        self.Y_obs = self.f_x[self.obs_int] + self.prng.normal(0, self.noise_sd.value, self.N_obs.value)
        self.update_plot()

    def fit_poly(self):
        bx = (np.polynomial.legendre.legvander(self.X_obs, self.complexity.value))
        self.mod = OLS(self.Y_obs, bx).fit()

    def polypredict(self, x):
        bx = (np.polynomial.legendre.legvander(x, self.complexity.value))
        return self.mod.predict(bx)

    def update_plot(self, _=None):
        self.fit_poly()
        f_hat = self.polypredict(self.X)
        y_hat = self.polypredict(self.X_obs)

        rmsq_error = np.sqrt(
            np.mean((f_hat - self.f_x) ** 2)
        )

        obs_rmsq_error = np.sqrt(
            np.mean((y_hat - self.Y_obs) ** 2)
        )

        with self.out:
            clear_output(wait=True)

            self.info.value = f"""
            <div style="
                font-size: 18px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
                width: 600px;
            ">
                <b>Root Mean Squared Error vs True Function:</b> {rmsq_error:.3f}<br>
                <b>Training RMSE:</b> {obs_rmsq_error:.3f}
            </div>
            """

            y_min = self.f_x.min()
            y_max = self.f_x.max()

            padding = 2 * self.noise_sd.value

            lower = y_min - padding
            upper = y_max + padding

            plt.figure(figsize=(12, 8))
            plt.scatter(self.X_obs, self.Y_obs, label="Observed data")
            plt.plot(self.X, f_hat, color="b", label="Estimated f(x)")
            plt.plot(self.X, self.f_x, color="r", label="True f(x)")
            plt.title(
                f"""N={self.N_obs.value}, Model complexity={self.complexity.value},Noise standard Deviation={self.noise_sd.value}""")
            plt.ylim(lower, upper)
            plt.ylim(max(lower, -3), min(upper, 3))  # optional global cap
            plt.legend()
            plt.show()

    def display(self):
        controls = widgets.VBox(
            [self.info, self.N_obs, self.complexity, self.noise_sd, self.button,
             self.true_button, ],
            layout=widgets.Layout(width="650px")
        )

        main = widgets.HBox(
            [controls, self.out],
            layout=widgets.Layout(width="100%")
        )

        display(main)
        self.generate_observations()
