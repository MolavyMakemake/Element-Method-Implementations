import PySimpleGUI as sg

menu_def = [
    ['&Select', ["Torus", ["FEM::TORUS", "VEM::TORUS"]]]
]

layout = [
    [sg.Menu(menu_def)],
    [sg.Text('Your window!', size=(30, 5))]
]


def main():
    window = sg.Window(title="window", layout=layout, margins=(100, 50))

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == "FEM::TORUS":
            pass
        else:
            print("Not implemented")

    window.close()


if __name__ == "__main__":
    main()