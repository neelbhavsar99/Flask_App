from flask import Flask, render_template, Response

application = Flask(__name__)

@application.route('/',methods=['GET','POST'])
def index():
    #return "Flask app is running"
    return render_template('index.html')

if __name__ == "__main__":
    application.run()