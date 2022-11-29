import os
import pandas as pd

import logging
from logging.handlers import TimedRotatingFileHandler


from flask import Flask, render_template, request, url_for, request, redirect, abort
from decision_maker import fit_predict

app = Flask(__name__)

name = __name__
FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = 'log.txt'
logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', encoding='utf-8')
file_handler.setFormatter(FORMATTER)

logger.addHandler(file_handler)

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_input(input):
    features = input.split(',')
    features_proc = {}

    for f in features:
        key, val = f.split(':')
        features_proc[key.strip()] = val.strip()

    return pd.DataFrame(pd.Series(features_proc)).T

def postprocess_output(outp):
    res = ['Если ']
    for step in outp:
        key, val = list(step.items())[0]
        if key == 'ответ':
            res.append('тогда ')
        if val == 'да':
            val = key.split('_')[-1]
            key = key.split(f'_{val}')[0]
        if val == 'нет':
            if key[-4:] == '_нет':
                key = key[:-4]
                val = 'да'
            if key[-3:] == '_да':
                key = key[:-3]
                val = 'нет'
        key = key.replace('_', ' ')
        res.append(f'{key} - {val}')
        if key != 'ответ':
            res.append(',<br>')
    return ''.join(res)

@app.route('/output', methods=['GET'])
def output():
    input = request.args.get('input')
    try:
        logger.info('---------------------')
        logger.info('Получено:')
        logger.info(f'{input}')
        input_proc = preprocess_input(input)
        res = fit_predict(input_proc)
        res = postprocess_output(res)
        logger.info('Выведено:')
        res_to_log = '\n' + res.replace("<br>", "\n")
        logger.info(f'{res_to_log}')
    except Exception as e:
        logger.warning(f'Вывод сделать нельзя')
        res = 'Вывод сделать нельзя'
    return res

if os.environ.get('APP_LOCATION') == 'heroku':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
else:
    app.run(host='localhost', port=8080, debug=True)