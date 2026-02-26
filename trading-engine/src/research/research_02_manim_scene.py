from __future__ import annotations

import pandas as pd
from manim import Axes, BLUE, GREEN, Create, DOWN, FadeIn, LEFT, Scene, Text, UP, VGroup


class Research02MarginalScene(Scene):
    def construct(self) -> None:
        data = pd.read_csv("artifacts/research_02/returns_timeseries.csv")
        base_curve = (1 + data["baseline"]).cumprod().to_numpy()
        combo_curve = (1 + data["combined"]).cumprod().to_numpy()

        n_points = len(base_curve)
        x_vals = [i / (n_points - 1) * 10 for i in range(n_points)]
        y_max = max(base_curve.max(), combo_curve.max())

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0.8, float(y_max) * 1.05, 0.2],
            x_length=10,
            y_length=5,
            axis_config={"include_numbers": False},
        )

        title = Text("Research 02: Marginal Value of Candidate Model", font_size=28).to_edge(UP)

        base_graph = axes.plot_line_graph(x_vals, base_curve, add_vertex_dots=False, line_color=BLUE)
        combo_graph = axes.plot_line_graph(x_vals, combo_curve, add_vertex_dots=False, line_color=GREEN)

        labels = VGroup(
            Text("Blue: Baseline Trading Engine", font_size=22, color=BLUE),
            Text("Green: Baseline + Candidate Model", font_size=22, color=GREEN),
        ).arrange(direction=DOWN, aligned_edge=LEFT).next_to(axes, direction=DOWN)

        self.play(FadeIn(title))
        self.play(Create(axes))
        self.play(Create(base_graph))
        self.play(Create(combo_graph))
        self.play(FadeIn(labels))
        self.wait(1.5)
