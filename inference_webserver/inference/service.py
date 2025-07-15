from flask import (Flask,
                   jsonify, 
                   flash, 
                   request, 
                   redirect, 
                   url_for)

app = Flask(__name__)

@app.route('/v1/inference/test', methods=['GET'])
def display_msg():
    return "welcome to isic inference"
