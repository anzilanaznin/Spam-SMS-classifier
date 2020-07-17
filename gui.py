from tkinter import *
window = Tk()
window.title("My app")
def browsefunc():
    filename = filedialog.askopenfilename()
    pathlabel.config(text=filename)
    print(filename)

browsebutton = Button(text="Browse", command=browsefunc)
browsebutton.pack()
pathlabel = Label()
pathlabel.pack()
window.geometry("400x400")
window.mainloop()