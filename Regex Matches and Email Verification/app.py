# app.py
from flask import Flask, render_template, request
import re

app = Flask(__name__)

def regex_matcher(test_string, regex_pattern):
    matches = re.findall(regex_pattern, test_string)
    return matches

def is_valid_email(email):
    # Regex pattern for a simple email validation
    email_pattern = r'^[\w\.-]+@[a-zA-Z\d\.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/regex_results', methods=['POST'])
def regex_results():
    test_string = request.form.get('test_string')
    regex_pattern = request.form.get('regex_pattern')

    regex_matches = regex_matcher(test_string, regex_pattern)

    return render_template('index.html', test_string=test_string, regex_pattern=regex_pattern,
                           regex_matches=regex_matches)



@app.route('/email_validation', methods=['POST'])
def email_validation():
    email_to_validate = request.form.get('email')

    is_valid_email_result = is_valid_email(email_to_validate)

    return render_template('index.html', email=email_to_validate,
                           is_valid_email=is_valid_email_result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')