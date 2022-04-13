import PySimpleGUI as sg
import os.path
import asyncio
import object_detection_camera_old as odc


async def main():
    file_list = []
    processed_file_list = []
    model_dir = ''

    file_select_column = [
        [
            sg.Text("Model Directory"),
            sg.Input(key='model_dir', size=(50, 1), do_not_clear=True),
            sg.FolderBrowse(key='MODEL_DIR_BROWSE'),
            sg.Button("Add", key='-MODEL_DIR_ADD-'),
        ],
        [
            sg.Text("Select images"),
            sg.Input(key="-FILE_SELECT-", size=(50, 1)),
            sg.FileBrowse(size=(30, 1), key="-SELECTED_IMAGE-"),
            sg.Button("Add", key="-ADD-"),
        ],
        [
            sg.Listbox(values=file_list, size=(50, 10), key="-FILE_LIST-"),
        ],
        [
            sg.Button("Process", key="-PROCESS-"),
        ]
        ]


    # ----- Full layout -----
    layout = [
        [
            sg.Column(file_select_column),
        ]
    ]

    window = sg.Window("Image Viewer", layout)

    # Run the Event Loop
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "-MODEL_DIR_ADD-":
            model_dir = values['MODEL_DIR_BROWSE']
            print(model_dir)

        if event == "-ADD-":
            file_list.append(values["-SELECTED_IMAGE-"])
            window['-FILE_LIST-'].Update(file_list)
            print(file_list)

        if event == "-PROCESS-":
            for file in file_list:
                processed_file_list.append(odc.object_detection(model_dir=model_dir, videocapture=file))
    window.close()

if __name__ == "__main__":
    asyncio.run(main())