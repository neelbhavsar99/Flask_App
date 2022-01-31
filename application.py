from flask import Flask

application = Flask(__name__)

@application.route('/',methods=['GET','POST'])
def index():
    return "Flask app is running"

if __name__ == "__main__":
    application.run()