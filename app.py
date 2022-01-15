import os
import cv2
import uuid
import sqlite3
import itertools
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


ASSETS_DIR = 'assets/'
DESCRIPTORS_DIR = ASSETS_DIR + 'features_vectors/'
DATABASE_PATH = ASSETS_DIR + 'matcher.db'


model = ResNet50(weights='imagenet')

ft_vec_cache = []



@app.route("/index", methods=["POST"])
@cross_origin()
def index():
    	
	if 'image' not in request.files:
		return jsonify({"data": "", "error": "'image' field is required.", "success": 0})

	if request.files['image'].filename == '':
		return jsonify({"data": "", "error": "'image' file is empty.", "success": 0})

	if request.files['image'].filename.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
		return jsonify({"data": "", "error": "'image' file must be either png or jpg or jpeg.", "success": 0})

	if 'product_id' not in request.form:
		return jsonify({"data": "", "error": "'product_id' field is required.", "success": 0})

	if 'product_name' not in request.form:
		return jsonify({"data": "", "error": "'product_name' field is required.", "success": 0})

	# if not request.form['product_id'].isdigit():
	# 	return jsonify({"data": "", "error": "'product_id' must be valid integer number.", "success": 0})


	# get image data
	image_name = request.files['image'].filename
	product_name = request.form['product_name']
	product_id = request.form['product_id']


	# extract features vector
	image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
	image = np.array(image)
	image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	features_vector = model.predict(image)[0]


    # save features_vector_file.
	features_vector_file_name = f"{uuid.uuid4().hex}.npy"
	file_path = DESCRIPTORS_DIR + features_vector_file_name
	with open(file_path, 'wb') as f:
		np.save(f, features_vector)


	# save data in db
	connection = sqlite3.connect(DATABASE_PATH)
	query = "INSERT INTO features_vectors (features_file_name, image_name, product_name, product_id) VALUES (?, ?, ?, ?)"
	connection.execute(query, [features_vector_file_name, image_name, product_name, product_id])
	connection.commit()
	connection.close()

	ft_vec_cache.append((product_id, product_name, features_vector))
	# print(f'size vector in index: {len(ft_vec_cache)}')

	return jsonify({"data": "indexed succesfully!", "error": "", "success": 1})
    		



@app.route("/search")
def search():
    	
	global ft_vec_cache
	# print(f'size vector in search: {len(ft_vec_cache)}')


	if 'image' not in request.files:
		return jsonify({"data": "", "error": "'image' field is required.", "success": 0})

	if request.files['image'].filename == '':
		return jsonify({"data": "", "error": "'image' file is empty.", "success": 0})

	if request.files['image'].filename.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
		return jsonify({"data": "", "error": "'image' file must be either png or jpg or jpeg.", "success": 0})
	

	# extract features vector
	image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
	image = np.array(image)
	image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	query_image_features_vector = model.predict(image)[0]


	matched_products = []


	if len(os.listdir(DESCRIPTORS_DIR)) != len(ft_vec_cache):
		connection = sqlite3.connect(DATABASE_PATH)
		rows = connection.execute("SELECT * FROM features_vectors").fetchall()
		connection.close()
		ft_vec_cache = []
		for _, features_file_name, image_name, product_name, product_id in rows:
			file_path = DESCRIPTORS_DIR + features_file_name
			with open(file_path, 'rb') as f:
				features_vector = np.load(f)
				similarity_score = np.linalg.norm(features_vector - query_image_features_vector)
				result = {"product_id": product_id, "product_name": product_name, "similarity_score": float(1 - similarity_score)}
				matched_products.append(result)
				ft_vec_cache.append((product_id, product_name, features_vector))
	else:
		for product_id, product_name, features_vector in ft_vec_cache:
			similarity_score = np.linalg.norm(features_vector - query_image_features_vector)
			result = {"product_id": product_id, "product_name": product_name, "similarity_score": float(1 - similarity_score)}
			matched_products.append(result)
    		

	matched_products = sorted(matched_products, key=lambda x: x['similarity_score'], reverse=True)[:4]

	response = jsonify({"data": matched_products, "error": "", "success": 1})
	response.headers.add("Access-Control-Allow-Origin", "*")
	return response



if __name__ == "__main__":
    app.run(host= '0.0.0.0')
