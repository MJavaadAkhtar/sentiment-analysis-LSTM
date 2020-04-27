from flask import Flask, render_template, url_for, request, render_template, jsonify
import json
# from loader import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single', methods=['POST', 'GET'])
def single():
    if request.method == 'POST':
        data = request.form.get('single')
        # analysis = {'data': int(get_model_data(data))}
        # return jsonify(analysis)
        return "hello"
    else:
        return "Hello, World Single"

@app.route('/multiple', methods=['POST', 'GET'])
def multiple():
    if request.method == 'POST':
        data = request.form.get('multiple')
        # analysis = {'data': get_multiple_data(data)}
        # return jsonify(analysis)
        return 'hhello'
    else:
        return "Hello, World Single"
if __name__ == "__main__":
    app.run(debug=True)