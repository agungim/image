from tkinter import *
import numpy as np
import cv2
import sys
import PIL.Image as imge
import PIL.ImageTk as imtk

curPth = sys.path[0]
imgPth = curPth+'/gr_agung1.png'
tmpPth = curPth+'/temp_agung1.png'

ev = None
thikness = 20


def click(event):
    global ev, back, lbl
    if ev == None:
        ev = event
        return None

    im = cv2.imread(imgPth)
    #mask = cv2.cvtColor(im.copy()*0, cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(im.copy()*0, cv2.COLOR_BGR2GRAY)
    cv2.line(mask, pt1=(ev.x, ev.y), pt2=(event.x, event.y),
             color=(255, 0, 0), thickness=thikness)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    out = im.copy()
    out[np.where(mask == 0)] = 255

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    back = imtk.PhotoImage(image=imge.fromarray(out))
    lbl.config(image=back)

    # print(mask[np.where(mask == 255)])
    x, X = min(ev.x, event.x)-thikness//2, max(ev.x, event.x)+thikness//2
    y, Y = min(ev.y, event.y)-thikness//2, max(ev.y, event.y)+thikness//2
    cropped = mask[y:Y, x:X]
    print(cropped, cropped.shape)
    print(cropped)
    a = np.asarray(cropped)
    np.savetxt(curPth+"/agung1.csv", a, delimiter=",")
    cv2.imwrite(curPth+'/2d_line_area.png', cropped)
    ev = event


root = Tk()
back = PhotoImage(file=imgPth)
lbl = Label(root, image=back)
lbl.place(x=0, y=0)
root.bind('<Button-1>', lambda event: click(event))
root.mainloop()