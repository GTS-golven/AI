import os.path
from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
from PIL import ImageTk,Image


def main():
    root = Tk()
    frame = ttk.Frame(root, padding="3 3 12 12")
    frame.pack(side=LEFT, fill=BOTH, expand=True)
    framerechts = ttk.Frame(root, padding="3 3 12 12")
    framerechts.pack(side=RIGHT, fill=BOTH, expand=True)
    # Define all frames
    frame1 = ttk.Frame(frame, padding=10, borderwidth=5, relief="ridge")
    frame1.pack(side=TOP, anchor=NW)

    listboxframe = ttk.Frame(frame, padding=10)
    listboxframe.pack(fill=BOTH, side=TOP, expand=True, anchor=W)

    frame2 = ttk.Frame(listboxframe, padding=10, width=500, height=800)
    frame2.pack(fill=BOTH, side=LEFT, expand=True, anchor=W)

    frame3 = ttk.Frame(listboxframe, padding=10, width=500, height=800)
    frame3.pack(fill=BOTH, side=RIGHT, expand=True, anchor=E)

    frame4 = ttk.Frame(frame, padding=10, width=200, height=10)
    frame4.pack(side=TOP, anchor=NW)

    # Frame 1
    choosefileframe = ttk.Frame(frame1)
    choosefileframe.pack(side=TOP, anchor=NW)
    ttk.Label(choosefileframe, text='Choose a file').pack(side=LEFT)
    ttk.Button(choosefileframe, text='Choose', command=get_file).pack(side=LEFT)

    choosemodelframe = ttk.Frame(frame1)
    choosemodelframe.pack(side=TOP, anchor=NW)
    ttk.Label(choosemodelframe, text='Choose a model').pack(side=LEFT)
    ttk.Button(choosemodelframe, text='Choose', command=get_model).pack(side=LEFT)

    ttk.Label(frame1, text="model_path").pack(side=TOP, anchor=NW)

    # Frame ListBox
    global fileListBox
    fileListBox = Listbox(frame2, height=5)
    fileListBox.bind('<<ListboxSelect>>', onselect)

    scrollbar = ttk.Scrollbar(frame2, orient=VERTICAL, command=fileListBox.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    fileListBox.pack(side=LEFT, fill=BOTH, expand=True)

    global fileListBox2
    fileListBox2 = Listbox(frame3, height=5)
    fileListBox2.pack(side=LEFT, fill=BOTH, expand=True)

    scrollbar2 = ttk.Scrollbar(frame3, orient=VERTICAL, command=fileListBox2.yview)
    scrollbar2.pack(side=RIGHT, fill=Y)

    # Frame 4
    ttk.Label(frame4, text='Process files').pack(side=LEFT)
    ttk.Button(frame4, text='Process', command=process_files).pack(side=LEFT)

    ttk.Label(frame4, text="Hello World!").pack(side=LEFT)
    ttk.Button(frame4, text="Quit", command=root.destroy).pack(side=LEFT)


    # Frame picture
    canvas = Canvas(framerechts, width=640, height=640)
    canvas.pack()
    global outputImage
    img = ImageTk.PhotoImage(Image.open(os.path.join(os.getcwd(), "Output", outputImage)))
    canvas.create_image(20, 20, anchor=NW, image=img)

    root.mainloop()


def get_file():
    root = Tk()
    root.withdraw()
    file_paths = fd.askopenfilenames(parent=root, title='Choose a file')

    for file in file_paths:
        fileListBox.insert(END, file)

    root.destroy()


def get_model():
    root = Tk()
    root.withdraw()

    global model_path
    model_path = fd.askdirectory(parent=root, title='Choose a model')
    print(model_path)

    root.destroy()


def process_files():
    try:
        # import object_detection_camera as odc
        print(f"Model selected {model_path}")
        # for file in fileListBox.get(0, END):
        #     try:
        #         odc.object_detection(model_path, file)
        #         fileListBox2.insert(END, file)
        #     except Exception as e:
        #         print(e)

        for file in fileListBox.get(0, END):
            fileListBox2.insert(END, os.path.join(os.getcwd(), "output", file.split("/")[-1]))

    except Exception as e:
        print(e)
    return


def onselect(evt):
    w = evt.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    print(value)


if __name__ == '__main__':
    main()
