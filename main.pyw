#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox
from PyQt5.QtWidgets import QDialog
from PyQt5 import uic

import sympy
from sympy.parsing import sympy_parser

import sys
import re
import logging
from traceback import format_exception
from numpy import *
from matplotlib.pyplot import *
PRECISION_LEVEL = 0.00001


class MyWindow(QMainWindow):
    CMD_BUFF_MAX_LEN = 25
    CMD_PREFIX = '<b>⇒</b>'
    # Константы для парсинга выражений
    Assign_pattern = re.compile('^[a-zA-Z0-9_]* += +.*')
    Plt_pattern = 'plots('
    Ltx_pattern = 'latex('
    LTr_pattern = 'laplace_transform('
    LTr_params = ", t, s, noconds=True"  # для работы laplace_transform
    InvLTr_pattern = 'inverse_laplace_transform('
    InvLTr_params = ", s, t"  # для работы inverse_laplace_transform
    ZTr_pattern = "z_transform("

    # Notes about patterns:
    # - Use '\*\*' as exponentiation operator (sympy use it)
    # - Escape python's regex's special symbols: '\(', '\)', '\+'
    num_pat = "(pi|E|\d+\.?\d*)"
    # 1/(s-+a) -> z/(z-exp(+-aT)
    ZTr_tabble1 = re.compile("1/\(s [+-] " + num_pat + "\)")
    # s/(s^2 + b^2) -> ( z^2 - z*cos(bT) ) / ( z^2 - 2*z*cos(bT) + 1 )
    ZTr_tabble_cos = re.compile("s/\(s\*\*2 \+ " + num_pat + "\)")
    # b/(s^2 + b^2) -> z*sin(bT) / ( z^2 - 2*z*cos(bT) + 1 )
    ZTr_tabble_sin = re.compile("1/\(s\*\*2 \+ " + num_pat + "\)")
    # b/((s+a)^2) + b^2) ->
    #         z*exp(-aT)*sin(bT) / (z^2 - 2*z*exp(-aT)*cos(bT) + exp(-2aT))
    # Note:
    # 1. 'b' will be removed as constant multiplier in
    #   table_forward_z_transform()
    # 2. sympy will expand denominator to s^2 + 2*a*s + (b^2+a^2)
    ZTr_tabble_esin = re.compile("1/\(s\*\*2 \+ " + num_pat + "\*s \+ " +
                                 num_pat + "\)")
    ZTr_tabble_esin_supplemental = re.compile(num_pat + "\*s")
    # (s+a) / ( (s+a)^2 + b^2 )
    ZTr_ecos_nom = "\(s \+ " + num_pat + "\)"
    ZTr_tabble_ecos = re.compile(ZTr_ecos_nom + "/\(s\*\*2 \+ " + num_pat +
                                 "\*s \+ " + num_pat + "\)")
    Excessive_zeroes = re.compile("\.0+")
    Excessive_ones1 = re.compile("^1\.0\*")
    Excessive_ones2 = re.compile(" 1\.0\*")
    Excessive_ones3 = re.compile("\(1\.0\*")

    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('ui/main.ui', self)

        self.input_area.returnPressed.connect(self.process_input)
        self.m_exit.triggered.connect(self.close)
        self.m_clear.triggered.connect(self.clear_cmd_buffer)
        self.m_tips.triggered.connect(self.show_help)
        self.output_area.anchorClicked.connect(self.remind_about_log)
        self.choose_mantisa_length.triggered.connect(self.customize_output)

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
        self.do_latex_output = False
        self.is_assign_action = False
        self.OUT_DIGIT_NUMBER = 4

        self.show()

    def process_input(self):
        """ Приложение реагирует на пользовательскую команду """

        self.input_area.selectAll()
        text = self.input_area.text()
        result = None

        if text.count('(') != text.count(')'):
            raise ValueError("Количество круглых скобок не совпадает!")

        if re.search(self.Assign_pattern, text):
            result = self.assign_action(text)
            self.is_assign_action = True
        elif text in self.variables:
            result = self.variables[text]
        elif text.startswith("orig("):
            inner = text[5:-1]
            expr, num_of_nodes = inner.split(',')
            expr = self.parse_expr(expr)
            result = self.calc_n_points_of_original(expr, int(num_of_nodes))
        elif text.startswith(self.Plt_pattern):
            result = text
            self.assigns_browser.append(result)
            text = text[6:-1].strip()
            new_text = text.split(',')
            self.assigns_browser.append(str(new_text))

            if len(new_text) == 1:
                self.plots(new_text[0])
            else:
                self.plots(new_text[0], new_text[1])
        elif text.startswith(self.Ltx_pattern):
            text = text[6:-1].strip()
            self.do_latex_output = True
            result = self.parse_expr(text)
        else:
            # Other command
            result = self.parse_expr(text)

        self.output(result)

    def parse_expr(self, expr):
        """ Приводим пользовательскую команду к приемлемому SymPy виду """
        # laplace_transform preparation
        if expr.startswith(self.LTr_pattern):
            expr = expr[:-1] + self.LTr_params + ')'

        # inverse_laplace_transform preparation
        if expr.startswith(self.InvLTr_pattern):
            expr = expr[:-1] + self.InvLTr_params + ')'

        # Z-transform
        if expr.startswith(self.ZTr_pattern):
            i = len(self.ZTr_pattern)
            expr = expr[i:-1]
            if ',' in expr:
                k = expr.index(',')
                discretization = expr[k + 1:]
                expr = expr[:k]
                expr = self.forward_z_transform(expr)
                expr = '(' + expr + ').subs(T, ' + discretization + ')'
            else:
                expr = self.forward_z_transform(expr)

        # use Python's exponentiation operator
        expr = re.sub(r"\^", "**", expr)
        expr = re.sub("δ", "KroneckerDelta", expr)

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
        cmd_end = "), s), s).evalf(" + str(self.OUT_DIGIT_NUMBER) + ")"
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

    def strip_mul_one(self, expr):
        expr = re.sub(self.Excessive_ones1, '', expr)
        expr = re.sub(self.Excessive_ones2, ' ', expr)
        expr = re.sub(self.Excessive_ones3, '(', expr)
        return expr

    def strip_zeroes(self, expr):
        """ Вырезать дробный хвост, если там одни нули """
        expr = re.sub(self.Excessive_zeroes, '', expr)
        return expr

    def strip_digits(self, expr):
        """ Обрезать дробный хвост до self.OUT_DIGIT_NUMBER """
        pat = '(\.\d{' + str(self.OUT_DIGIT_NUMBER) + '})\d+'
        pat = re.compile(pat)

        return re.sub(pat, r'\1', expr)

    def table_forward_z_transform(self, expr):
        """
        expr - объект SymPy
        Сверяем с эталонными значениями и, если совпало, преобразовываем по шаблону
        """
        expr, expr_coef = self.prepare_table_expression(expr)

        res = ""
        expr = self.strip_mul_one(str(expr))
        expr = self.strip_zeroes(expr)
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
            coef = expr[10:-1]  # get number; 10: '1/(s**2 + '

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
            b_coef = expr[expr.rindex('+') + 2:-1]

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

    def output(self, sympy_obj):
        output = self.strip_zeroes(str(sympy_obj))
        output = self.strip_digits(output)
        output = re.sub(r"\*\*", "^", output)  # '^' for exponentiation
        output = re.sub("KroneckerDelta", 'δ', output)

        self.print_output(output)

        if self.do_latex_output:
            self.latex_output(self.strip_zeroes(sympy.latex(sympy_obj)))
            self.do_latex_output = False
        elif self.is_assign_action:
            self.assigns_browser.clear()
            valT = str(self.variables.get('T'))
            if valT != 'T':
                self.assigns_browser.append("T = " + valT)

            for (var, val) in self.variables.items():
                txt = str(val)
                txt = self.strip_mul_one(txt)
                txt = self.strip_zeroes(txt)
                txt = self.strip_digits(txt)
                txt = re.sub(r"\*\*", "^", txt)  # '^' for exponentiation
                txt = re.sub("KroneckerDelta", 'δ', txt)
                if var != 'z' and var != 't' and var != 's' and var != 'T':
                    self.assigns_browser.append(var + " = " + txt)
            self.is_assign_action = False

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

    def customize_output(self):
        dlg = QDialog()
        uic.loadUi('ui/preferences.ui', dlg)
        dlg.exec_()
        if dlg.result() == 1:
            self.OUT_DIGIT_NUMBER = dlg.num_of_digits.value()

    def latex_output(self, output):
        text(0, 0.6, "${}$".format(output), fontsize=24)
        fig = gca()
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        draw()  # or savefig
        show()

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

    def calc_n_points_of_original(self, expr, n):
        if n <= 0:
            raise ValueError("Пытаемся посчитать оригинал для {} точек".format(n))

        tail = 0
        z = self.variables['z']
        INF = sympy.oo
        prev = sympy.limit(expr, z, INF)
        res = [prev]
        for i in range(n-1):
            tail += z**(-i) * prev
            calc_it = sympy.simplify(z**(i+1) * (expr - tail))
            prev = sympy.limit(calc_it, z, INF)
            res.append(prev)

        return res

    def show_help(self):
        help_you = QDialog()
        uic.loadUi('ui/help.ui', help_you)
        help_you.exec_()

    def give_data_array(self, data, expr):
        t = sympy.symbols('t')
        data_array = []
        for i in data:
            element = expr.subs({t: i})
            data_array.append(element)
        return data_array

    def plots(self, expr, expr2=False, discretization=False):
        if discretization:  # Получить из N точек expr с шагом discretization
            pass
        else:               # Просто рисовать непрерывный граффик
            plotting_data = arange(0.0, 5.0, 0.5)

            first_expr = expr
            second_expr = expr2
            self.assigns_browser.append(str(first_expr))
            first_parse = sympy_parser.parse_expr(first_expr)
            first_data_array = self.give_data_array(plotting_data, first_parse)
            if second_expr is False:
                pass
            else:
                self.assigns_browser.append(str(second_expr))
                second_parse = sympy_parser.parse_expr(second_expr)
                second_data_array = self.give_data_array(plotting_data, second_parse)
                legend_second_text = str(second_expr)
                plot(first_data_array, '-', label=legend_first_text, color='blue')
                plot(second_data_array, 'ro', label=legend_second_text, color='red')

            legend_first_text = str(first_expr)
            fig = figure()
            ax = fig.add_subplot(111)
            grid(True)
            plot(first_data_array)
            legend(loc='best')
            ax.set_title(legend_first_text)
            show()


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
