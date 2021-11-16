from tkinter import *

from createWindow import CreateWindow


def main():
    root = Tk()
    root.geometry("500x500")
    CreateWindow(root)


if __name__ == "__main__":
    main()