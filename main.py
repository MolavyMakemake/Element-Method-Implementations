import PySimpleGUI as sg

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

import FEM
import VEM
import plot

matplotlib.use('TkAgg')


menu_def = [
    ["&solve", ["&poisson", "&spectrum"]],
    ["&domain", ["&rectangle", "&elliptic disk", "&torus"]],
    ['&method', ["&finite elements", "&virtual elements"]],
]

layout = [
    [sg.Menu(menu_def)],
    [sg.Text("Finite elements on Poisson with Dirichlet b.c.")],

    [sg.Text("Bounds: "),
     sg.Text("("), sg.Input("-1.0", s=(4, None), key="inp_x0", pad=0), sg.Text(",", pad=0),
     sg.Input("-1.0", s=(4, None), key="inp_y0", pad=0), sg.Text(")"), sg.Text("x", pad=0),
     sg.Text("("), sg.Input("1.0", s=(4, None), key="inp_x1", pad=0), sg.Text(",", pad=0),
     sg.Input("1.0", s=(4, None), key="inp_y1", pad=0), sg.Text(")"),

     sg.Text(";   Resolution: "),
     sg.Input("21", s=(2, None), key="inp_res0", pad=0), sg.Text("x", pad=0),
     sg.Input("21", s=(2, None), key="inp_res1", pad=0)],

    [sg.Text("f(z) =", key="txt_f"), sg.Input("np.exp(-20 * z * np.conj(z))", key="inp_f"),
     sg.Button("view eigenvectors", key="btn_eigen", visible=False),
     sg.Checkbox("fix trace", key="fix trace", default=True, visible=False, enable_events=True)],

    [sg.HorizontalSeparator()],
    [sg.Checkbox("complex plot", key="cplot", default=False, enable_events=True)],
    [sg.Canvas(key="solution_plot"), sg.Canvas(key="matrix_plot")],
]


def main():
    window = sg.Window(title="window", layout=layout, margins=(100, 50), finalize=True)

    window["inp_x0"].bind("<Return>", "")
    window["inp_y0"].bind("<Return>", "")
    window["inp_x1"].bind("<Return>", "")
    window["inp_y1"].bind("<Return>", "")
    window["inp_res0"].bind("<Return>", "")
    window["inp_res1"].bind("<Return>", "")
    window["inp_f"].bind("<Return>", "")

    rplot_fig = plt.figure(0)
    cplot_fig = plt.figure(1)
    rplot_ax = rplot_fig.subplots(subplot_kw={"projection": "3d"})
    cplot_ax = cplot_fig.subplots()
    matrix_fig = plt.figure(2)
    matrix_ax = matrix_fig.subplots()

    solution_canvas = window["solution_plot"].TKCanvas
    matrix_canvas = window["matrix_plot"].TKCanvas
    solution_figCanvasAgg = FigureCanvasTkAgg(rplot_fig, solution_canvas)
    matrix_figCanvasAgg = FigureCanvasTkAgg(matrix_fig, matrix_canvas)

    solution_figCanvasAgg.get_tk_widget().pack(side='top', fill='both', expand=1)
    matrix_figCanvasAgg.get_tk_widget().pack(side='top', fill='both', expand=1)

    model = FEM.Model()
    u = model.solve_poisson(lambda z: np.exp(-20 * z * np.conj(z)))

    u = np.real(u)
    plot.surface(rplot_ax, model.vertices, model.triangles, u)
    plot.add_wireframe(rplot_ax, model.vertices, model.polygons, u)
    matrix_ax.matshow(model.L)
    solution_figCanvasAgg.draw()
    matrix_figCanvasAgg.draw()

    solve = "poisson"
    eigen_i = 0
    isPlotComplex = False
    while True:
        event, values = window.read()
        print("event:", event)

        if event == sg.WIN_CLOSED:
            break
        elif event in ["poisson", "spectrum"]:
            solve = event
            window["txt_f"].update(visible=(event == "poisson"))
            window["inp_f"].update(visible=(event == "poisson"))
            window["btn_eigen"].update(visible=(event == "spectrum"))
            window["fix trace"].update(visible=(event == "spectrum"))

            model.computeSpectrumOnBake = event == "spectrum"
            if event == "poisson" and not model.isTraceFixed:
                model.isTraceFixed = True
                model.bake()

            if event == "spectrum":
                model.bake_spectrum()

        elif event == "finite elements":
            model = FEM.Model(model.domain, model.bounds, model.resolution, model.isTraceFixed, model.computeSpectrumOnBake)
        elif event == "virtual elements":
            model = VEM.Model(model.domain, model.bounds, model.resolution, model.isTraceFixed, model.computeSpectrumOnBake)

        elif event in ["rectangle", "elliptic disk", "torus"]:
            model.domain = event
            model.bake()
        elif event in ["inp_res0", "inp_res1"]:
            try:
                model.resolution[0] = int(values["inp_res0"])
                model.resolution[1] = int(values["inp_res1"])
            except ValueError:
                print("Invalid resolution")

            model.bake()
        elif event == "btn_eigen":
            popup_layout = [[sg.Column(
                [[sg.Text(f"{model.eigenvalues[i]:.5f}"), sg.Button("view", key=str(i))]
                 for i in range(len(model.eigenvalues))],
                scrollable=True, size=(500, 700))]]
            popup_window = sg.Window(title="Visualize eigenvector", layout=popup_layout)
            popup_event, popup_values = popup_window.read()
            if popup_event is not None:
                eigen_i = int(popup_event)
            popup_window.close()

        elif event == "fix trace":
            model.isTraceFixed = values["fix trace"]
            model.bake()

        elif event == "cplot":
            isPlotComplex = values[event]

            solution_figCanvasAgg.get_tk_widget().forget()
            if isPlotComplex:
                solution_figCanvasAgg = FigureCanvasTkAgg(cplot_fig, solution_canvas)
            else:
                solution_figCanvasAgg = FigureCanvasTkAgg(rplot_fig, solution_canvas)
            solution_figCanvasAgg.get_tk_widget().pack(side='top', fill='both', expand=1)

        try:
            model.bounds = np.array([
                [float(values["inp_x0"]), float(values["inp_x1"])],
                [float(values["inp_y0"]), float(values["inp_y1"])]
            ])
        except ValueError:
            print("Invalid bounds")
            continue

        try:
            if solve == "poisson":
                u = model.solve_poisson(lambda z: eval(values["inp_f"]))
            elif solve == "spectrum":
                u = model.eigenvectors[eigen_i, :]

            if isPlotComplex:
                cplot_ax.clear()
                plot.complex(cplot_ax, model.vertices, model.triangles, u)
                plot.add_wireframe(cplot_ax, model.vertices, model.polygons)
            else:
                rplot_ax.clear()
                u = np.real(u)
                plot.surface(rplot_ax, model.vertices, model.triangles, u)
                plot.add_wireframe(rplot_ax, model.vertices, model.polygons, u)

            matrix_ax.matshow(model.L)
            matrix_figCanvasAgg.draw()

            solution_figCanvasAgg.draw()

        except Exception as e:
            print(e)
            continue

    window.close()


if __name__ == "__main__":
    main()