import PySimpleGUI as sg

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

import FEM
import plot

matplotlib.use('TkAgg')


menu_def = [[
    "&select", [
        '&method', ["&finite elements", "&virtual elements"],
        "&domain", ["&rectangle", "&elliptic disk", "&torus"]
    ]]
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

    [sg.Text("f(z) ="), sg.Input("np.exp(-20 * z * np.conj(z))", key="inp_f")],

    [sg.Canvas(key="canvas")],
]


def main():
    window = sg.Window(title="window", layout=layout, margins=(100, 50),
                        finalize=True)


    window["inp_x0"].bind("<Return>", "")
    window["inp_y0"].bind("<Return>", "")
    window["inp_x1"].bind("<Return>", "")
    window["inp_y1"].bind("<Return>", "")
    window["inp_res0"].bind("<Return>", "")
    window["inp_res1"].bind("<Return>", "")
    window["inp_f"].bind("<Return>", "")

    ax = plt.figure().add_subplot(projection='3d')
    figure = plt.gcf()
    canvas = window["canvas"].TKCanvas

    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

    model = FEM.Model()
    x, y, u = model.solve_poisson(lambda z: np.exp(-20 * z * np.conj(z)))

    while True:
        plt.cla()
        plot.surface(x, y, model.polygons, u, ax)
        figure_canvas_agg.draw()

        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == "finite elements":
            pass
        elif event == "virtual elements":
            pass
        elif event in ["rectangle", "elliptic disk", "torus"]:
            model.domain = event
            model.bake()
            print(event)
        elif event in ["inp_res0", "inp_res1"]:
            try:
                model.resolution[0] = int(values["inp_res0"])
                model.resolution[1] = int(values["inp_res1"])
            except ValueError:
                print("Invalid resolution")

            model.bake()

        try:
            model.bounds = np.array([
                [float(values["inp_x0"]), float(values["inp_x1"])],
                [float(values["inp_y0"]), float(values["inp_y1"])]
            ])
        except ValueError:
            print("Invalid bounds")

        try:
            x, y, u = model.solve_poisson(lambda z: eval(values["inp_f"]))
        except:
            print("Could not solve")
            continue

    window.close()


if __name__ == "__main__":
    main()