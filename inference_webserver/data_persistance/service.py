from flask import (Flask,
                   jsonify, 
                   flash, 
                   request, 
                   redirect, 
                   url_for)

app = Flask(__name__)

@app.route('/v1/data-persistance/test', methods=['GET'])
def display_msg():
    return "welcome to isic data-persistance"
