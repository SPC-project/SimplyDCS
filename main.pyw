#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox
from PyQt5.QtWidgets import QDialog
from PyQt5 import uic

import matplotlib.pyplot as plt
import sympy
from sympy.parsing import sympy_parser

import sys
import re
import logging
from traceback import format_exception

PRECISION_LEVEL = 0.00001


class MyWindow(QMainWindow):
    CMD_BUFF_MAX_LEN = 25
    CMD_PREFIX = '<b>⇒</b>'
    # Константы для парсинга выражений
    Plt_pattern = re.compile('^plot\(.*\)')
    LTr_pattern = re.compile('^laplace_transform\(.*\)')
    LTr_params = ", t, s, noconds=True"  # для работы laplace_transform
    InvLTr_pattern = re.compile('inverse_laplace_transform\(.*\)')
    InvLTr_params = ", s, t"  # для работы inverse_laplace_transform
    ZTr_name = "z_transform"
    ZTr_offset = len(ZTr_name)
    ZTr_pattern = re.compile("^" + ZTr_name + "\(.*\)")

    # Notes about patterns:
    # - Use '\*\*' as exponentiation operator (sympy use it)
    # - Escape python's regex's special symbols: '\(', '\)', '\+'
    num_pat = "(pi|E|\d+\.\d+)"
    # 1/(s-+a) -> z/(z-exp(+-aT)
    ZTr_tabble1 = re.compile("1\.0/\(s [+-] " + num_pat + "\)")
    # s/(s^2 + b^2) -> ( z^2 - z*cos(bT) ) / ( z^2 - 2*z*cos(bT) + 1 )
    ZTr_tabble_cos = re.compile("s/\(s\*\*2 \+ " + num_pat + "\)")
    # b/(s^2 + b^2) -> z*sin(bT) / ( z^2 - 2*z*cos(bT) + 1 )
    ZTr_tabble_sin = re.compile("1\.0/\(s\*\*2 \+ " + num_pat + "\)")
    # b/((s+a)^2) + b^2) ->
    #         z*exp(-aT)*sin(bT) / (z^2 - 2*z*exp(-aT)*cos(bT) + exp(-2aT))
    # Note:
    # 1. 'b' will be removed as constant multiplier in
    #   table_forward_z_transform()
    # 2. sympy will expand denominator to s^2 + 2*a*s + (b^2+a^2)
    ZTr_tabble_esin = re.compile("1\.0/\(s\*\*2 \+ " + num_pat + "\*s \+ " +
                                 num_pat + "\)")
    ZTr_tabble_esin_supplemental = re.compile(num_pat + "\*s")
    # (s+a) / ( (s+a)^2 + b^2 )
    ZTr_ecos_nom = "\(s \+ " + num_pat + "\)"
    ZTr_tabble_ecos = re.compile(ZTr_ecos_nom + "/\(s\*\*2 \+ " + num_pat +
                                 "\*s \+ " + num_pat + "\)")
    Excessive_zeroes = re.compile("\.0+")

    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('ui/main.ui', self)

        self.input_area.returnPressed.connect(self.process_input)
        self.m_exit.triggered.connect(self.close)
        self.m_clear.triggered.connect(self.clear_cmd_buffer)
        self.m_tips.triggered.connect(self.show_help)
        self.output_area.anchorClicked.connect(self.remind_about_log)

        self.variables = {
            't': sympy.Symbol(
                't', positive=True),
            'T': sympy.Symbol(
                'T', positive=True),
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

        if text.count('(') != text.count(')'):
            raise ValueError("Количество круглых скобок не совпадает!")

        if '=' in text:  # TODO: будут ошибки в выражениях со сравнением: "=="
            result = self.assign_action(text)
        elif text in self.variables:
            result = self.variables[text]
        elif re.search(self.Plt_pattern, text):
            text = text[5:-1]
            self.plot(text)
        else:
            # Other command
            result = self.parse_expr(text)

        self.output(result)

    def parse_expr(self, expr):
        """ Приводим пользовательскую команду к приемлемому SymPy виду """
        # laplace_transform preparation
        match = re.search(self.LTr_pattern, expr)
        if match:
            i = match.end() - 1
            expr = expr[:i] + self.LTr_params + expr[i:]

        # inverse_laplace_transform preparation
        match = re.search(self.InvLTr_pattern, expr)
        if match:
            i = match.end() - 1
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
        expr = text[name_len + 1:].strip()

        if name not in self.variables:
            self.variables[name] = sympy.var(name)

        self.variables[name] = self.parse_expr(expr)

        return name + " = " + str(self.variables[name])

    def forward_z_transform(self, to_transform_expr):
        """ Разложить на простые множители и каждый заменить по таблице """
        cmd_start = "apart(collect(simplify("
        cmd_end = "), s), s).evalf()"
        expr = self.parse_expr(cmd_start + to_transform_expr + cmd_end)
        res = ""
        if isinstance(expr, sympy.Add):
            for summand in expr.args:
                if summand.is_number:  # deal with constants
                    raise ValueError("Получили константу в z-преобразовании: " +
                                     str(expr))

                res += self.table_forward_z_transform(summand)
        else:
            res = self.table_forward_z_transform(expr)

        if res[1] == '+':
            res = res[2:]  # remove lead '+'

        return res

    def prepare_table_expression(self, expr):
        if not isinstance(expr, sympy.Mul):
            return expr, 1

        expr_coef = 1
        for arg in expr.args:
            if arg.is_number:  # It should be only one numerical coefficient
                if abs(arg - 1.0) > PRECISION_LEVEL:
                    expr = expr / arg
                    expr_coef = arg

        return expr, expr_coef

    def table_forward_z_transform(self, expr):
        """
        expr - объект SymPy
        Сверяем с эталонными значениями и, если совпало, преобразовываем по шаблону
        """
        expr, expr_coef = self.prepare_table_expression(expr)

        res = ""
        expr = str(expr)
        if expr == "1/s":
            res = "z/(z-1)"
        elif expr == "s**(-2)" or expr == "1/s**2":
            res = "T*z/(z-1)^2"
        elif expr == "s**(-3)" or expr == "1/s**3":
            res = "T^2*z*(z+1)/(z-1)^3"
        elif self.ZTr_tabble1.fullmatch(expr):
            coef = expr[5:-1]
            # Swap sign because table says to do it
            if coef[0] == '+':
                coef = "-" + coef[1:]
            else:
                coef = "+" + coef[1:]
            res += "z/(z-exp(" + coef + "*T))"
        elif self.ZTr_tabble_cos.fullmatch(expr):
            coef = expr[10:-1]  # get number; 10: length of 's/(s**2 + '
            coef = 'sqrt(' + str(coef) + ')'
            cos_ = "cos(" + coef + "*T)"
            res = "(z^2 - z*" + cos_ + ") / (z^2 - 2*z*" + cos_ + " + 1)"
        elif self.ZTr_tabble_sin.fullmatch(expr):
            coef = expr[12:-1]  # get number; 10: '1.0/(s**2 + '
            if coef == "1" and expr_coef == 1:
                res = "z*sin(T) / (z^2 - 2*z*cos(T) + 1)"
            elif expr_coef:
                coef = sympy.sqrt(float(coef))

                expr_coef = self.check_coef(expr_coef, coef)

                b = str(expr_coef)
                res = "z*sin(" + b + "*T) "
                res += "/ (z^2 - 2*z*cos(" + b + "*T) + 1)"
        elif self.ZTr_tabble_esin.fullmatch(expr):
            pos = re.search(self.ZTr_tabble_esin_supplemental, expr)
            a_coef = float(expr[pos.start():pos.end() - 2]) / 2.0
            b_coef = sympy.sqrt(float(expr[pos.end() + 3:-1]) - a_coef**2)

            expr_coef = self.check_coef(expr_coef, b_coef)

            a_coef = str(a_coef)
            b_coef = str(b_coef)
            nominator = "z*exp(-{}*T)*sin({}*T)"
            denominator = "z^2 - 2*z*exp(-{}*T)*cos({}*T) + exp(-2*{}*T)"
            res = nominator.format(a_coef, b_coef) + "/ ("
            res += denominator.format(a_coef, b_coef, a_coef) + ")"
        elif self.ZTr_tabble_ecos.fullmatch(expr):
            a_coef = expr[5:expr.index(')')]  # len('(s + ')
            b_coef = expr[expr.rindex('+')+2:-1]

            arg_a = a_coef + "*T"
            a_coef = float(a_coef)
            b_coef = float(b_coef)
            b_coef = sympy.sqrt(b_coef - a_coef**2)
            arg_b = str(b_coef) + "*T"
            nominator = "(z^2 - z*exp(-{})*cos({}))"
            denominator = "(z^2 - 2*z*exp(-{})*cos({}) + exp(-2*{}))"
            res = nominator.format(arg_a, arg_b) + "/ ("
            res += denominator.format(arg_a, arg_b, arg_a) + ")"
        else:
            raise ValueError("Нет правила для преобразования: " + expr)

        if expr_coef == 1:
            res = " + " + res
        elif expr_coef == -1:
            res = " - " + res
        else:
            if expr_coef < 0:
                res = str(expr_coef) + "*" + res
            else:
                res = " + " + str(expr_coef) + "*" + res

        return res

    def check_coef(self, nom_coef, denom_coef):
        delta = abs(nom_coef - denom_coef)
        if delta < PRECISION_LEVEL:
            return nom_coef

        return nom_coef / denom_coef

    def strip_zeroes(self, expr):
        expr = re.sub(self.Excessive_zeroes, '', expr)
        return expr

    def output(self, sympy_obj):
        # TODO LaTeX here
        plt.text(0, 0.6, "${}$".format(sympy.latex(sympy_obj)), fontsize=30)
        fig = plt.gca()
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # plt.draw()  # or savefig
        # plt.show()

        output = self.strip_zeroes(str(sympy_obj))
        output = re.sub(r"\*\*", "^", output)  # '^' for exponentiation
        self.print_output(output)

    def print_output(self, text):
        self.cmd_buffer += "<p>" + self.CMD_PREFIX + " " + text + "</p>"
        self.cmd_buffer_len += 1
        if self.cmd_buffer_len > self.CMD_BUFF_MAX_LEN:
            mov1 = 6  # length of '<p><b>' - need to maintain cmd_buffer
            mov2 = 3  # length of '\p>
            second_cmd_start = self.cmd_buffer[mov1:].index(self.CMD_PREFIX)
            self.cmd_buffer = self.cmd_buffer[second_cmd_start + mov2:]
            self.cmd_buffer_len = self.CMD_BUFF_MAX_LEN

        self.output_area.setText(self.cmd_buffer)
        it = self.output_area.verticalScrollBar()
        it.setValue(it.maximum())

    def clear_cmd_buffer(self):
        self.cmd_buffer = ""
        self.cmd_buffer_len = 0
        self.output_area.setText(self.cmd_buffer)

    def shit_happens(self, msg):
        self.print_output("<a href='#log_remind'><i>Ошибка!</i></a> — " + msg)
        self.statusbar.addWidget(self.error_desc)

    def remind_about_log(self):
        msg = QMessageBox(QMessageBox.Critical, "Ошибка!",
                          "Данные об ошибке записаны в файл errors.log")
        msg.exec_()

    def show_help(self):
        help_you = QDialog()
        uic.loadUi('ui/help.ui', help_you)
        help_you.exec_()

    def plot(self, expr):
        pass


def handel_exceptions(type_, value, tback):
    """
    Перехватывает исключения, логгирует их и не позволяет уронить программу
    """
    error_msg = ''.join(format_exception(type_, value, tback))
    error_expr = window.input_area.text()
    logging.error(error_msg + 'Current expression: ' + error_expr + '\n')
    sys.__excepthook__(type_, value, tback)
    last_line = error_msg[:-1].rindex("\n")
    window.shit_happens(error_msg[last_line + 1:-1])


if __name__ == '__main__':
    log_format = '[%(asctime)s]  %(message)s'
    logging.basicConfig(
        format=log_format, level=logging.ERROR, filename='errors.log')

    app = QApplication(sys.argv)
    window = MyWindow()

    sys.excepthook = handel_exceptions
    sys.exit(app.exec_())
