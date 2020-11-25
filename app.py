from flask import Flask, request, jsonify
from heisenberg.Matcher import Matcher

app = Flask(__name__)



@app.route("/index", methods=["POST"])
def index():
    	
	if 'image' not in request.files:
		return jsonify({"data": "", "error": "'image' field is required.", "success": 0})

	if request.files['image'].filename == '':
		return jsonify({"data": "", "error": "'image' file is empty.", "success": 0})

	if request.files['image'].filename.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
		return jsonify({"data": "", "error": "'image' file must be either png or jpg or jpeg.", "success": 0})

	if 'product_id' not in request.form:
		return jsonify({"data": "", "error": "'product_id' field is required.", "success": 0})

	if not request.form['product_id'].isdigit():
		return jsonify({"data": "", "error": "'product_id' must be valid integer number.", "success": 0})

	if Matcher.fit(request.form['product_id'], request.files['image']):
		return jsonify({"data": "indexed succesfully!", "error": "", "success": 1})
	else:
		return jsonify({"data": "", "error": "'product_id' field is required.", "success": 0})
    		



@app.route("/search")
def search():
	if 'image' not in request.files:
		return jsonify({"data": "", "error": "'image' field is required.", "success": 0})

	if request.files['image'].filename == '':
		return jsonify({"data": "", "error": "'image' file is empty.", "success": 0})

	if request.files['image'].filename.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
		return jsonify({"data": "", "error": "'image' file must be either png or jpg or jpeg.", "success": 0})
	
	prediction = Matcher.predict(request.files['image'])
	
	if prediction != False:
		return jsonify({"data": prediction, "error": "", "success": 1})
	else:
		return jsonify({"data": "", "error": "No products found.", "success": 0})



if __name__ == "__main__":
    app.run(host= '0.0.0.0')
