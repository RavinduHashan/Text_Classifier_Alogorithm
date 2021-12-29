from flask import Flask, request, jsonify
import Prediction

app = Flask(__name__)

@app.route('/dataset', methods=["GET", "POST"])
def getData():
    if request.method == 'GET':
        return jsonify("This is get method")
    else:
        reviews = request.json['reviews']
        itemType = request.json['item_type']
        Prediction.prediction(reviews,itemType)
        return jsonify("Saving data is successfully")

if __name__ == '__main__':
    app.run(debug=True)