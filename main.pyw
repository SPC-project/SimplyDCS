#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
import sympy
from sympy.parsing.sympy_parser import parse_expr


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('ui/main.ui', self)

        self.input_area.returnPressed.connect(self.handle_input)

        self.show()

    def handle_input(self):
        text = self.input_area.text()
        t = sympy.symbols('t', positive = True)
        s = sympy.symbols('s')
        self.output_area.setText(sympy.pretty(parse_expr(text)))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())
