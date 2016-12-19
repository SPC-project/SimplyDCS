#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox
from PyQt5.QtWidgets import QDialog
from PyQt5 import uic

import sympy
from sympy.parsing import sympy_parser
from sympy import exp, sin, cos, pi, E, I

import sys
import re
import logging
from traceback import format_exception


class MyWindow(QMainWindow):
    CMD_BUFF_MAX_LEN = 25
    CMD_PREFIX = '<b>⇒</b>'
    # Константы для парсинга выражений
    LTr_pattern = re.compile('^laplace_transform\(.*\)')
    LTr_params = ", t, s, noconds=True"  # для работы laplace_transform
    InvLTr_pattern = re.compile('inverse_laplace_transform\(.*\)')
    InvLTr_params = ", s, t"  # для работы inverse_laplace_transform
    ZTr_name = "z_transform"
    ZTr_offset = len(ZTr_name)
    ZTr_pattern = re.compile("^" + ZTr_name + "\(.*\)")
    num_pat = "(pi|E|[0-9]+)"
    ZTr_tabble1 = re.compile("^1/\(s [+-] " + num_pat + "\)$")

    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('ui/main.ui', self)

        self.input_area.returnPressed.connect(self.process_input)
        self.m_exit.triggered.connect(self.close)
        self.m_clear.triggered.connect(self.clear_cmd_buffer)
        self.m_tips.triggered.connect(self.show_help)
        self.output_area.anchorClicked.connect(self.remind_about_log)

        self.variables = {
            't': sympy.Symbol('t', positive=True),
            'T': sympy.Symbol('T', positive=True),
            's': sympy.Symbol('s'),
            'z': sympy.Symbol('z')
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
            # Other command
            result = self.parse_expr(text)

        self.output(result)

    def parse_expr(self, expr):
        """ Приводим пользовательскую команду к приемлемому SymPy виду """
        # laplace_transform preparation
        match = re.search(self.LTr_pattern, expr)
        if match:
            i = match.end()-1
            expr = expr[:i] + self.LTr_params + expr[i:]

        # inverse_laplace_transform preparation
        match = re.search(self.InvLTr_pattern, expr)
        if match:
            i = match.end()-1
            expr = expr[:i] + self.InvLTr_params + expr[i:]

        # Z-transform
        match = re.search(self.ZTr_pattern, expr)
        if match:
            i = match.start() + self.ZTr_offset + 1  # +1 for '(' symbol
            j = match.end() - 1
            expr = self.forward_z_transform(expr[i:j])

        # use Python's exponentiation operator
        expr = re.sub(r"\^", "**", expr)

        return sympy_parser.parse_expr(expr, local_dict=self.variables)

    def assign_action(self, text):
        name_len = text.index('=')
        name = text[:name_len].strip()
        expr = text[name_len+1:].strip()

        if name not in self.variables:
            self.variables[name] = sympy.var(name)

        self.variables[name] = self.parse_expr(expr)

        return name + " = " + str(self.variables[name])

    def forward_z_transform(self, to_transform_expr):
        """ Разложить на простые множители и каждый заменить по таблице """
        expr = "apart(collect(simplify(" + to_transform_expr + "), s), s)"
        summands = self.parse_expr(expr)
        res = ""
        if isinstance(summands, sympy.Add):
            for summand in summands.args:
                res += self.table_forward_z_transform(summand)
        else:
            res = self.table_forward_z_transform(summands)

        if res[0] == '+':
            res = res[2:]  # remove lead '+'

        return res

    def table_forward_z_transform(self, expr):
        """ expr - объект SymPy """
        if expr.is_number:  # deal with constants
            return str(expr)

        expr = str(expr)
        res = " + "
        if expr[0] == '-':
            res = " - "
            expr = expr[1:]

        if expr == "1/s":
            res += "z/(z-1)"
        elif expr == "s**(-2)" or expr == "1/s**2":
            res += "T*z/(z-1)^2"
        elif expr == "s**(-3)" or expr == "1/s**3":
            res += "T^2*z*(z+1)/(z-1)^3"
        elif re.search(self.ZTr_tabble1, expr):
            coef = expr[5:-1]
            # Swap sign because table says to do it
            if coef[0] == '+':
                coef = "-" + coef[1:]
            else:
                coef = "+" + coef[1:]
            res += "z/(z-exp(" + coef + "*T))"
        else:
            raise ValueError("Can not transform expression: " + expr)

        return res

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
        msg = QMessageBox(QMessageBox.Critical, "Ошибка!",
                          "Данные об ошибке записаны в файл errors.log")
        msg.exec_()

    def show_help(self):
        help_you = QDialog()
        uic.loadUi('ui/help.ui', help_you)
        help_you.exec_()


def handel_exceptions(type_, value, tback):
    """
    Перехватывает исключения, логгирует их и не позволяет уронить программу
    """
    error_msg = ''.join(format_exception(type_, value, tback))
    error_expr = window.input_area.text()
    logging.error(error_msg + 'Current expression: ' + error_expr + '\n')
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
