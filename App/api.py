import os
import time
from flask import request, render_template, Flask

from main import plot_dispatcher
from PIL import Image
from flask import session

# session.clear()


import numpy as np

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder='templates')
app.config["DEBUG"] = True


def play_sound():
    duration = 0.4
    freq = 440
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


@app.route('/')
def home():
    get_industries_all()
    return render_template('entity-name-form.html')


@app.route('/submit')
def submit():
    pass


def get_industries(filename):
    """
    Parameters
    ----------
    filename : string
        the path for the html file that displays a select drop-down menu of all industries

    """
    file = open(filename, 'r')
    lines = []
    for line in file:
        lines.append(line)
    for i in range(len(lines)):
        if "<option" in lines[i]:
            lines[i] = "#\n"
    ind_to_comp = np.load(
        "templates/industry_to_companies.npy",
        allow_pickle=True)
    industries = list(ind_to_comp[()].keys())
    index = 0
    for i in range(len(lines)):
        if index <= len(industries) - 1 and lines[i] == "#\n":
            new_option = '''<option value ="''' + \
                industries[index] + '''">''' + industries[index] + '''</option>\n'''
            index += 1
            lines[i] = new_option
        elif index > 0 and index <= len(industries) - 1 and lines[i] != "#\n":
            new_option = '''<option value ="''' + \
                industries[index] + '''">''' + industries[index] + '''</option>\n'''
            index += 1
            lines.insert(i, new_option)

    html_content = ""
    for line in lines:
        if line[-1] != '\n':
            html_content += line[:-1]
            html_content += "\n"
        else:
            html_content += line
    file = open(filename, "w")
    file.write(html_content)
    file.close()


@app.route('/analysis-details', methods=['GET', 'POST'])
def form_post():
    if request.method == 'POST':
        dict_form = request.form
        get_industries_all()
        if dict_form['entity-type'] == 'industry':
            get_industries("templates/industry-option.html")
            return render_template("industry-option.html")
        elif dict_form['entity-type'] == 'company':
            get_industries("templates/company-option.html")
            return render_template('company-option.html')
        elif dict_form['entity-type'] == 'company-comparison':
            # get_industries("templates/company-comparison-option.html")
            return render_template('company-comparison-option.html')
        elif dict_form['entity-type'] == 'tweet-analysis':
            # get_industries("templates/tweet-analysis-option.html")
            return render_template('tweet-analysis-option.html')
        elif dict_form['entity-type'] == 'industry-change':
            get_industries("templates/industry-editing-option.html")
            return render_template('industry-editing-option.html')
        else:
            return render_template('error-page.html')
        get_industries_all()
        return dict_form


def handle_user_input(dict_form, entity_type):
    """
    Parameters
    ----------
    dict_form : ImmutableMultiDict
    entity_type : string
        either "company" or "industry"

    Returns
    ----------
    an HTML rendered file
    """
    start = time.clock()
    path = str(os.getcwd())
    if path[-3:] == 'App':
        os.chdir(os.path.dirname(os.getcwd()))
    COUNTER, bytes_image, number_of_tweets, entity_name = plot_dispatcher(
        dict_form)
    if bytes_image == 0:
        score = COUNTER
        if score > 0:
            return render_template(
                'tweet-analysis-report.html',
                score=score,
                label='positive')
        elif score == 0:
            return render_template(
                'tweet-analysis-report.html', score=score, label='')
        else:
            return render_template(
                'tweet-analysis-report.html', score=(-1) * score, label='negative')
    elif bytes_image == -1:
        return render_template('not-enough-tweets-page.html')
    elif bytes_image == -2:
        return render_template('error-page.html')
    image = Image.open(bytes_image)
    image.save(
        "App/static/" +
        entity_type +
        "-report-" +
        str(COUNTER) +
        ".png")
    number_of_seconds = 4 * round(time.clock() - start, 2)
    # play_sound()
    return render_template(
        "report-analysis.html",
        number_of_tweets=number_of_tweets,
        number_of_seconds=number_of_seconds,
        COUNTER=COUNTER,
        entity_type=entity_type,
        entity_name=entity_name)


@app.route('/company-option', methods=['GET', 'POST'])
def company_post():
    if request.method == 'POST':
        return handle_user_input(request.form, 'company')


@app.route('/industry-option', methods=['GET', 'POST'])
def industry_post():
    if request.method == 'POST':
        return handle_user_input(request.form, 'industry')


@app.route('/company-comparison-option', methods=['GET', 'POST'])
def company_comparison_post():
    if request.method == 'POST':
        return handle_user_input(request.form, 'company')


@app.route('/tweet-analysis-option', methods=['GET', 'POST'])
def tweet_analysis_post():
    if request.method == 'POST':
        return handle_user_input(request.form, 'company')


def delete_company(company, industry):
    """
    Parameters
    ----------
    company : string
        the name of the company to be deleted
    industry : string
        the industry that company is a part of
    """
    ind_to_comp = np.load(
        "templates/industry_to_companies.npy",
        allow_pickle=True)
    for company_name in ind_to_comp[()][industry]:
        if company == company_name:
            ind_to_comp[()][industry].remove(company)
            np.save("templates/industry_to_companies.npy", ind_to_comp)
            return 1
    return -1


def add_company(company, industry):
    """
    Parameters
    ----------
    company : string
        the name of the company to be added
    industry : string
        the industry that company is a part of
    """
    ind_to_comp = np.load(
        "templates/industry_to_companies.npy",
        allow_pickle=True)
    if delete_company(company, industry) == -1:
        ind_to_comp[()][industry].append(company)
    np.save("templates/industry_to_companies.npy", ind_to_comp)


def get_industries_all():
    path = str(os.getcwd())
    if path[-6:] == 'master':
        os.chdir(path + "/App")
    get_industries("templates/industry-editing-option.html")
    get_industries("templates/industry-option.html")
    get_industries("templates/company-option.html")


def delete_industry(industry):
    """
    Parameters
    ----------
    industry : string
        the industry to be deleted
    """
    ind_to_comp = np.load(
        "templates/industry_to_companies.npy",
        allow_pickle=True)
    if industry not in ind_to_comp[()].keys():
        return -1
    else:
        del ind_to_comp[()][industry]
    np.save("templates/industry_to_companies.npy", ind_to_comp)
    get_industries_all()


def add_industry(industry):
    """
    Parameters
    ----------
    industry : string
        the industry to be added
    """
    ind_to_comp = np.load(
        "templates/industry_to_companies.npy",
        allow_pickle=True)
    if industry in ind_to_comp[()].keys():
        return -1
    else:
        ind_to_comp[()][industry] = []
    np.save("templates/industry_to_companies.npy", ind_to_comp)
    get_industries_all()


def modify_industry(dict_form):
    """
    Parameters
    ----------
    dict_form : ImmutableMultiDict

    Returns
    ----------
    an HTML rendered file
    """
    industry = dict_form['industry-name']
    options = [
        dict_form['company-add'],
        dict_form['company-removal'],
        dict_form['industry-add'],
        dict_form['industry-removal']]
    completed_options_counter = 0
    for option in options:
        if len(option) != 0:
            completed_options_counter += 1
        if completed_options_counter == 2:
            return render_template(
                'error-page.html',
                text=" when you performed more than one editing action in the same time")
    modified = ""
    if dict_form['company-add'] != '':
        add_company(dict_form['company-add'], industry)
        modified = "company"
    elif dict_form['company-removal'] != '':
        delete_company(dict_form['company-removal'], industry)
        modified = "company"
    elif dict_form['industry-add'] != '':
        add_industry(dict_form['industry-add'])
        modified = "industry"
    elif dict_form['industry-removal'] != '':
        delete_industry(dict_form['industry-removal'])
        modified = "industry"
    path = str(os.getcwd())
    if path[-6:] == 'master':
        os.chdir(path + "/App")
    ind_to_comp = np.load(
        "templates/industry_to_companies.npy",
        allow_pickle=True)
    get_industries_all()
    if modified == 'company':
        page_title = industry.capitalize() + " companies"
    else:
        page_title = "Industries"
    html_content = """<!DOCTYPE html>
    <html>
    <head>
        <title>""" + page_title + """</title>
        <head>
        <style>
    div {
      border-radius: 5px;
      background-color: #f2f2f2;
      padding: 20px;
    }
        </style>
    </head>
    <body>
    <div>"""
    if modified == "company" or modified == "":
        html_content += """<h1>""" + industry + """ companies:</h1>"""
        html_content += "<ul>"
        for company in ind_to_comp[()][industry]:
            html_content += "<li>"
            html_content += company
            html_content += "</li>"
        html_content += "</ul>"
        html_content += "</div></body></html>"
        file = open("templates/companies-from-industry.html", "w")
        file.write(html_content)
        file.close()
        return render_template('companies-from-industry.html')
    elif modified == "industry":
        html_content += """<h1>Industries:</h1>"""
        html_content += "<ul>"
        for industry_item in ind_to_comp[()].keys():
            html_content += "<li>"
            html_content += industry_item
            html_content += "</li>"
        html_content += "</ul>"
        html_content += "</div></body></html>"
        file = open("templates/industries-list.html", "w")
        file.write(html_content)
        file.close()
        return render_template('industries-list.html')


@app.route('/industry-editing-option', methods=['GET', 'POST'])
def industry_editing_post():
    if request.method == 'POST':
        get_industries_all()
        return modify_industry(request.form)


if __name__ == '__main__':
    app.run()
