#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox
from PyQt5 import uic

import sympy
from sympy.parsing import sympy_parser

import sys
import re
import logging
from traceback import format_exception


class MyWindow(QMainWindow):
    LTr_pattern = re.compile('^laplace_transform\(.*\)')
    LTr_params = ", t, s, noconds=True"  # для работы laplace_transform
    InvLTr_pattern = re.compile('inverse_laplace_transform\(.*\)')
    InvLTr_params = ", s, t"  # для работы inverse_laplace_transform
    CMD_BUFF_MAX_LEN = 25
    CMD_PREFIX = '<b>⇒</b>'

    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('ui/main.ui', self)

        self.input_area.returnPressed.connect(self.process_input)
        self.m_exit.triggered.connect(self.close)
        self.m_clear.triggered.connect(self.clear_cmd_buffer)
        self.output_area.anchorClicked.connect(self.remind_about_log)

        self.variables = {
            't': sympy.Symbol('t', positive=True),
            's': sympy.Symbol('s'),
        }
        self.cmd_buffer = ""
        self.cmd_buffer_len = 0
        self.error_desc = QLabel("Произошла ошибка!")

        self.show()

    def process_input(self):
        """ Приложение реагирует на пользовательскую команду """
        self.input_area.selectAll()
        text = self.input_area.text()
        result = None

        if '=' in text:  # TODO: будут ошибки в выражениях со сравнением: "=="
            result = self.assign_action(text)
        elif text in self.variables:
            result = self.variables[text]
        else:
            result = self.parse_expr(text)

        self.output(result)

    def parse_expr(self, expr):
        """ Приводим пользовательскую команду к приемлемому SymPy виду """
        result = re.search(self.LTr_pattern, expr)
        if result:  # laplace_transform fixes
            print("work")
            i = result.end()-1
            expr = expr[:i] + self.LTr_params + expr[i:]

        result = re.search(self.InvLTr_pattern, expr)
        if result:  # inverse_laplace_transform fixes
            i = result.end()-1
            expr = expr[:i] + self.InvLTr_params + expr[i:]

        expr = re.sub(r"\^", "**", expr)  # Python's exponentiation operator

        return sympy_parser.parse_expr(expr, local_dict=self.variables)

    def assign_action(self, text):
        name_len = text.index('=')
        name = text[:name_len].strip()
        expr = text[name_len+1:].strip()

        if name not in self.variables:
            self.variables[name] = sympy.var(name)

        self.variables[name] = self.parse_expr(expr)

        return name + " = " + str(self.variables[name])

    def output(self, sympy_obj):
        # TODO LaTeX here

        output = str(sympy_obj)
        output = re.sub(r"\*\*", "^", output)  # '^' for exponentiation
        self.print_output(output)

    def print_output(self, text):
        self.cmd_buffer += "<p>" + self.CMD_PREFIX + " " + text + "</p>"
        self.cmd_buffer_len += 1
        if self.cmd_buffer_len > self.CMD_BUFF_MAX_LEN:
            mov1 = 6  # length of '<p><b>' - need to maintain cmd_buffer
            mov2 = 3  # length of '\p>
            second_cmd_start = self.cmd_buffer[mov1:].index(self.CMD_PREFIX)
            self.cmd_buffer = self.cmd_buffer[second_cmd_start+mov2:]
            self.cmd_buffer_len = self.CMD_BUFF_MAX_LEN

        self.output_area.setText(self.cmd_buffer)

    def clear_cmd_buffer(self):
        self.cmd_buffer = ""
        self.cmd_buffer_len = 0
        self.output_area.setText(self.cmd_buffer)

    def shit_happens(self):
        self.print_output("<a href='#log_remind'><i>Ошибка!</i></a>")
        self.statusbar.addWidget(self.error_desc)

    def remind_about_log(self):
        print("Here")
        msg = QMessageBox(QMessageBox.Critical, "Ошибка!",
                          "Данные об ошибке записаны в файл errors.log")
        msg.exec_()


def handel_exceptions(type_, value, tback):
    """
    Перехватывает исключения, логгирует их и не позволяет уронить программу
    """
    logging.error(''.join(format_exception(type_, value, tback)))
    sys.__excepthook__(type_, value, tback)
    window.shit_happens()


if __name__ == '__main__':
    log_format = '[%(asctime)s]  %(message)s'
    logging.basicConfig(format=log_format, level=logging.ERROR,
                        filename='errors.log')

    app = QApplication(sys.argv)
    window = MyWindow()

    sys.excepthook = handel_exceptions
    sys.exit(app.exec_())
