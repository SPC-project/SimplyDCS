SimplyDCS
===============

###### (проект разрабатывается студентами кафедры [СПУ](http://www.kpispu.info/ru/about) [НТУ "ХПИ"](http://www.kpi.kharkov.ua/ru/))

##### Функционал:
- **laplace_transform**(*expr*) — применить преобразование Лапласа к *expr*
- **inverse_laplace_transform**(*expr*) — обратное преобразование Лапласа
- **z_transform**(*expr*) — Z-преобразование
- **latex**(*expr*) — отобразить *expr* средствами LaTeX
- **plot**(*expr*) — построить график для *expr*
- **symplify**(*expr*) — упростить *expr*
    + simplify( *sin(t)^2 + cos(t)^2* ) ⇒ 1
    + simplify( (x^3 + x^2 - x - 1)/(x^2 + 2\*x + 1) ) ⇒ x-1
- **expand**(*expr*) — раскрыть скобки в *expr*
- **factor**(*expr*) — "складывает" полином
    + factor( *x^3 - x^2 + x - 1* ) ⇒ (x-1)*(x^2+1)
- **collect**(*expr*, *var*) — "собирает" коэффициенты при одинаковых степенях *var*
- **apart**(*expr*) — разложить на дроби
- *expr*.**evalf**() — вычислить значение *expr*
    + ( E + pi + exp(2) ).evalf() ⇒ 13.2489305809795
- **integrate**(*expr*, *var*) — символьное интегрирование
    + integrate(cos(x), x) ⇒ sin(x)
- **integrate**(*expr*, (*var*, *lower limit*, *upper limit*))
    + integrate(exp(-x), (x, 0, oo)) ⇒ 1

##### Запуск
Для работы программы необходимы (в скобках - версии ПО, которые используются при отладке):
- Интерпретатор для Python 3 (3.5)
- Qt 5 (5.6)
- PyQt 5 (5.6)
- matplotlib (1.5.3)
- simpy (1.0)
- python-tk (3.5)
- **Anaconda (4.2.0) - если Вам лень устанавливать библиотеки**

Если перечисленные пакеты установлены, запустите файл **main.pyw**
