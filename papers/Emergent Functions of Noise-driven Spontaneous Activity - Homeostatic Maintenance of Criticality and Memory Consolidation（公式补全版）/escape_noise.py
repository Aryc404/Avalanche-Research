"""Escape noise: ρ(t) = (1/τ) exp[(v(t) − v_th)/b].

左图显示发放强度 ρ(v)，右图显示固定膜电位下截至 t 至少一次放电的概率。
"""

import math
import tkinter as tk
from tkinter import ttk


def escape_rate_hz(voltage_mv, tau_ms, b_mv, threshold_mv):
    # τ 从毫秒换成秒，使 ρ 的单位为 Hz。
    return 1000.0 / tau_ms * math.exp((voltage_mv - threshold_mv) / b_mv)


def firing_probability(rate_hz, time_ms):
    # 恒定膜电位下 ρ 不随时间改变；Hz × ms / 1000 为无量纲。
    return -math.expm1(-rate_hz * time_ms / 1000.0)


class EscapeNoisePlot:
    def __init__(self, root):
        self.root = root
        root.title("Escape noise 公式可视化")
        root.geometry("1280x760")
        root.minsize(1050, 640)

        panels = ttk.Frame(root)
        panels.pack(fill="both", expand=True, padx=16, pady=(16, 6))
        panels.columnconfigure(0, weight=1, uniform="plots")
        panels.columnconfigure(1, weight=1, uniform="plots")
        panels.rowconfigure(2, weight=1)
        for column, formula, description in [
            (0, "ρ(t) = (1/τ) · exp[(v(t) − v_th) / b]",
             "瞬时发放强度随膜电位变化"),
            (1, "1 − S(t) = 1 − exp[−∫₀ᵗ ρ(s) ds]",
             "右图固定膜电位：1 − S(t) = 1 − exp[−ρ(v)t]"),
        ]:
            ttk.Label(panels, text=formula, font=("Cambria", 18)).grid(
                row=0, column=column, pady=(0, 6)
            )
            ttk.Label(panels, text=description).grid(
                row=1, column=column, pady=(0, 10)
            )

        self.canvas = tk.Canvas(panels, background="white", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        self.prob_canvas = tk.Canvas(panels, background="white", highlightthickness=0)
        self.prob_canvas.grid(row=2, column=1, sticky="nsew", padx=(8, 0))
        for canvas in (self.canvas, self.prob_canvas):
            canvas.bind("<Configure>", lambda event: self.draw())
        ttk.Label(
            root, text="右图表示 [0, t] 内至少一次放电的累计概率。单位换算：ρ[Hz] × t[ms] / 1000。"
        ).pack(pady=(0, 4))

        controls = ttk.Frame(root, padding=(24, 6, 24, 16))
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        self.values = {}
        self.readouts = {}
        settings = [
            ("tau", "τ：时间尺度 (ms)", 1.0, 50.0, 16.8448675),
            ("b", "b：电压尺度 (mV)", 1.0, 10.0, 4.0),
            ("threshold", "v_th：阈值参数 (mV)", -65.0, -45.0, -54.0),
            ("voltage", "v：左图选取膜电位 (mV)", -85.0, -35.0, -64.0),
            ("time", "右图时间范围 (ms)", 10.0, 5000.0, 500.0),
        ]
        for row, (key, label, lower, upper, initial) in enumerate(settings):
            value = tk.DoubleVar(value=initial)
            self.values[key] = value
            ttk.Label(controls, text=label, width=24).grid(
                row=row, column=0, sticky="w", pady=5
            )
            ttk.Scale(
                controls, from_=lower, to=upper, variable=value,
                command=lambda _value: self.draw(),
            ).grid(row=row, column=1, sticky="ew", padx=12)
            readout = ttk.Label(controls, width=10, anchor="e")
            readout.grid(row=row, column=2)
            self.readouts[key] = readout

        self.draw()

    def draw(self):
        if len(self.readouts) != 5:
            return
        tau = self.values["tau"].get()
        b = self.values["b"].get()
        threshold = self.values["threshold"].get()
        voltage = self.values["voltage"].get()
        time_max = self.values["time"].get()
        for key, label in self.readouts.items():
            label.config(text=f"{self.values[key].get():.3f}")

        # 左图标记与右图计算共用这一个强度值。
        selected_rate = escape_rate_hz(voltage, tau, b, threshold)
        self.draw_rate(tau, b, threshold, voltage, selected_rate)
        self.draw_probability(voltage, selected_rate, time_max)

    def draw_rate(self, tau, b, threshold, voltage, selected_rate):
        c = self.canvas
        c.delete("all")
        width, height = c.winfo_width(), c.winfo_height()
        if width < 200 or height < 160:
            return
        left, right = 83, width - 22
        top, bottom = 84, height - 62
        v_min, v_max = -85.0, -35.0
        rho_max = escape_rate_hz(v_max, tau, b, threshold) * 1.08

        def x_position(voltage):
            return left + (voltage - v_min) / (v_max - v_min) * (right - left)

        def y_position(rate):
            return bottom - rate / rho_max * (bottom - top)

        font = ("Microsoft YaHei", 9)
        for i in range(6):
            rate = rho_max * i / 5
            y = y_position(rate)
            c.create_line(left, y, right, y, fill="#e5e9ec")
            text = f"{rate:.3g}" if rate < 10000 else f"{rate:.2e}"
            c.create_text(left - 10, y, text=text, anchor="e", font=font)
        for tick_voltage in range(-85, -34, 10):
            x = x_position(tick_voltage)
            c.create_line(x, top, x, bottom, fill="#e5e9ec")
            c.create_text(x, bottom + 17, text=str(tick_voltage), font=font)

        c.create_line(left, top, left, bottom, right, bottom, fill="#334155")
        c.create_text(left, 15, text="瞬时发放强度 ρ(t) (Hz)", anchor="w", font=font)
        c.create_text(left, 37, anchor="w", font=font, fill="#c56526",
                      text=f"选中：v = {voltage:.2f} mV → ρ(v) = {selected_rate:.4g} Hz",
                      tags="selected_values")
        c.create_text((left + right) / 2, height - 17,
                      text="膜电位 v(t) (mV)", font=font)

        threshold_x = x_position(threshold)
        c.create_line(threshold_x, top, threshold_x, bottom,
                      fill="#75818c", dash=(5, 4))
        c.create_text(threshold_x - 6, top + 14,
                      text=f"v_th = {threshold:.2f} mV", anchor="e",
                      fill="#75818c", font=font)

        points = []
        for i in range(501):
            curve_voltage = v_min + (v_max - v_min) * i / 500
            rate = escape_rate_hz(curve_voltage, tau, b, threshold)
            points.extend((x_position(curve_voltage), y_position(rate)))
        c.create_line(*points, fill="#087f8c", width=2.5)

        # 标出 v=v_th 时的强度，显示它是有限值 1/τ。
        threshold_rate = 1000.0 / tau
        threshold_y = y_position(threshold_rate)
        c.create_oval(threshold_x - 4, threshold_y - 4,
                      threshold_x + 4, threshold_y + 4,
                      fill="#087f8c", outline="")
        c.create_text(left, 59, anchor="w", font=font, fill="#75818c",
                      text=f"ρ(v_th) = 1/({tau:.3f} ms) = {threshold_rate:.3f} Hz")

        # 同色标记连接左图选定的电位与右图的概率曲线。
        selected_x = x_position(voltage)
        selected_y = y_position(selected_rate)
        c.create_line(selected_x, top, selected_x, bottom,
                      fill="#c56526", width=1.5, dash=(4, 3),
                      tags="selected_voltage_line")
        c.create_line(left, selected_y, selected_x, selected_y,
                      fill="#c56526", width=1.5, dash=(4, 3),
                      tags="selected_rate_line")
        c.create_oval(selected_x - 6, selected_y - 6,
                      selected_x + 6, selected_y + 6,
                      fill="#c56526", outline="white", width=1.5,
                      tags="selected_point")

    def draw_probability(self, voltage, rate, time_max):
        c = self.prob_canvas
        c.delete("all")
        width, height = c.winfo_width(), c.winfo_height()
        if width < 200 or height < 160:
            return
        left, right = 62, width - 26
        top, bottom = 84, height - 62
        font = ("Microsoft YaHei", 9)

        def x_position(time):
            return left + time / time_max * (right - left)

        def y_position(probability):
            return bottom - probability * (bottom - top)

        for i in range(6):
            probability = i / 5
            y = y_position(probability)
            c.create_line(left, y, right, y, fill="#e5e9ec")
            c.create_text(left - 10, y, text=f"{probability:.1f}", anchor="e", font=font)
            time = time_max * i / 5
            x = x_position(time)
            c.create_line(x, top, x, bottom, fill="#e5e9ec")
            c.create_text(x, bottom + 17, text=f"{time:.3g}", font=font)

        c.create_line(left, top, left, bottom, right, bottom, fill="#334155")
        c.create_text(left, 15, text="累计放电概率 1 − S(t) (无量纲)", anchor="w", font=font)
        c.create_text(left, 37, anchor="w", font=font, fill="#c56526",
                      text=f"取左图：v = {voltage:.2f} mV → ρ(v) = {rate:.4g} Hz",
                      tags="selected_values")
        c.create_text(left, 59, anchor="w", font=font, fill="#75818c",
                      text="用此 ρ 计算：1 − S(t) = 1 − exp[−ρ · t / 1000]")
        c.create_text((left + right) / 2, height - 17, text="时间 t (ms)", font=font)

        # 在高强度时补充短时间采样点，避免首次跃升被画成一段长斜线。
        times = {time_max * i / 500 for i in range(501)}
        fast_max = min(time_max, 8000.0 / rate)
        times.update(fast_max * i / 300 for i in range(301))
        points = []
        for time in sorted(times):
            probability = firing_probability(rate, time)
            points.extend((x_position(time), y_position(probability)))
        c.create_line(*points, fill="#c56526", width=2.5, tags="probability_curve")

        end_probability = firing_probability(rate, time_max)
        c.create_text(left + 12, bottom - 55, anchor="nw", font=font,
                      text=f"t = {time_max:.3g} ms 时，1 − S(t) = {end_probability:.4f}",
                      fill="#c56526")


if __name__ == "__main__":
    window = tk.Tk()
    EscapeNoisePlot(window)
    window.mainloop()
