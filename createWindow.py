from tkinter import *
import threading
import time

import numpy as np
from PIL import Image, ImageOps
import io

import Recovering



frame_width = 1000
frame_height = 1000
canvas_width = 200
canvas_height = 200
threadActive = False


class CreateWindow:

    def paint(self, event):
        x1, y1 = (event.x - 3), (event.y - 3)
        x2, y2 = (event.x + 3), (event.y + 3)
        self.w.create_rectangle(x1, y1, x2, y2, fill="#000000", width="1", outline="#000000")
        global threadActive
        if not threadActive:
            m = DiscoverThread("", self)
            m.start()
            threadActive = True


    def lunchReconize(self):
        ps = self.w.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img_rsize=img.resize((150,150))
        img_rsize.save('img.jpeg')

        predict = Recovering.reconize()
        formatter = "{0:.2f}"
        prediction = np.argmax(predict), ":", formatter.format(predict[0][np.argmax(predict)] * 100), "%"
        print(prediction)
        res = ""
        if np.argmax(predict) == 0:
            res = "Etoile"
        elif np.argmax(predict) == 1:
            res = "Lune"
        elif np.argmax(predict) == 2:
            res = "Nuage"
        elif np.argmax(predict) == 3:
            res = "Parapluie"
        elif np.argmax(predict) == 4:
            res = "Soleil"

        self.label["text"] = res, ":", formatter.format(predict[0][np.argmax(predict)] * 100), "%"

    def resetdraw(self):
        self.w.delete("all")
        self.w.create_rectangle(0, 0, 500, 500, fill="#FFFFFF", width="1", outline="#000000")
        self.label["text"] = "???"

    def __init__(self, master):
        master.title("Painting")

        self.f = Frame(master,
                       width=frame_width,
                       height=frame_height,
                       bg='#FF5733'
                       )
        self.f.pack(expand=YES, fill=BOTH)

        self.w = Canvas(self.f,
                        width=canvas_width, height=canvas_height, bg="#FFFFFF")
        #self.w.create_rectangle(0,0,canvas_width,canvas_height,fill="#000000", width="1", outline="#000000")
        self.w.pack(expand=NO)
        self.w.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.w.bind("<B1-Motion>", self.paint)

        self.button2 = Button(self.f, text="Press to reset drawn", command=self.resetdraw)
        self.button2.pack(side=BOTTOM)

        self.button = Button(self.f, text="Press to discover", command=self.lunchReconize)
        self.button.pack(side=BOTTOM)

        self.label = Label(self.f, text="???")
        self.label.pack(side=BOTTOM)


        mainloop()


class DiscoverThread (threading.Thread):
    def __init__(self, nom = '', param = ''):
        threading.Thread.__init__(self)
        self.nom = nom
        self.param = param
        self._stopevent = threading.Event()

    def run(self):
        time.sleep(1)
        i = 0
        while not self._stopevent.isSet():
            m = CreateWindow
            m.lunchReconize(self.param)
            self.stop()

            self._stopevent.wait(1.0)
        print("le thread s'est termine proprement")
        global threadActive
        threadActive = False

    def stop(self):
        self._stopevent.set()