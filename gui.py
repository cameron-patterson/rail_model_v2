from tkinter import *
from e_field_analysis import e_field
import matplotlib.pyplot as plt

conditions = ["Dry", "Moderate", "Wet"]
sections = ["Glasgow to Edinburgh", "Preston to Lancaster"]
sect = "glasgow_edinburgh_falkirk"
cond = "dry"


def submit_ex():
    print("Ex: " + str(scale_ex.get()) + " V/km")
    global ex
    ex = scale_ex.get()


def submit_ey():
    print("Ey: " + str(scale_ey.get()) + " V/km")
    global ey
    ey = scale_ey.get()


def set_condition():
    global cond
    if x.get() == 0:
        cond = conditions[0].lower()
    if x.get() == 1:
        cond = conditions[1].lower()
    if x.get() == 2:
        cond = conditions[2].lower()


def set_section():
    global sect
    if y.get() == 0:
        sect = "glasgow_edinburgh_falkirk"
    if y.get() == 1:
        sect = "preston_lancaster"


def click():
    try:
        ia, ib = e_field(exs=[ex], eys=[ey], section_name=sect, conditions=cond)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ia[0], '.')
        ax[0].axhline(0.055, c="red")
        ax[0].axhline(0.081, c="green")
        ax[1].plot(ib[0], '.')
        ax[1].axhline(0.055, c="red")
        ax[1].axhline(0.081, c="green")
        plt.show()
    except NameError as error:
        print("Please ensure you have clicked submit to lock in your electric field values")


window = Tk()
window.geometry("600x300")
window.title("Signalling Network Model")

icon = PhotoImage(file="train.png")
window.iconphoto(True, icon)
window.config(background="black")

frame1 = Frame(window, bg="white", bd=5)
frame1.place(x=0, y=0)
frame2 = Frame(window, bg="white", bd=5)
frame2.place(x=120, y=0)
frame3 = Frame(window, bg="white", bd=5)
frame3.place(x=240, y=0)
frame4 = Frame(window, bg="white", bd=5)
frame4.place(x=240, y=120)
frame5 = Frame(window, bg="white", bd=5)
frame5.place(x=420, y=160)


# Ex scale
scale_ex = Scale(frame1, from_=-10, to=10, length=200, width=20, resolution=0.1, tickinterval=2)
scale_ex.pack()
button_scale_ex = Button(frame1, text="Submit", command=submit_ex, bd=5)
button_scale_ex.pack()
label_ex = Label(frame1, text="Ex")
label_ex.pack()

# Ey scale
scale_ey = Scale(frame2, from_=-10, to=10, length=200, width=20, resolution=0.1, tickinterval=2)
scale_ey.pack()
button_scale_ey = Button(frame2, text="Submit", command=submit_ey, bd=5)
button_scale_ey.pack()
label_ey = Label(frame2, text="Ey")
label_ey.pack()

# Condition selector
x = IntVar()
for index in range(len(conditions)):
    radiobutton = Radiobutton(frame4, text=conditions[index], variable=x, value=index, font=("Times New Roman", 20),
                              command=set_condition, bg="white", pady=5)
    radiobutton.pack()

# Line selector
y = IntVar()
for indey in range(len(sections)):
    radiobutton = Radiobutton(frame3, text=sections[indey], variable=y, value=indey, font=("Times New Roman", 20),
                              command=set_section, bg="white", pady=5)
    radiobutton.pack()

# Run button
button_run = Button(frame5, text="Run", command=click, font=("Times New Roman", 30), fg="#00FF00", bg="black",
                    activeforeground="#00FF00", activebackground="black", state=ACTIVE)
button_run.pack()

window.mainloop()
