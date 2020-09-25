import os
import numpy as np
import uuid
import cv2
import sqlite3

class Matcher():

	ASSETS_DIR = 'assets/'
	DESCRIPTORS_DIR = ASSETS_DIR + 'descriptors/'
	DATABASE_PATH = ASSETS_DIR + 'matcher.db'


	@classmethod
	def fit(cls, product_id, image_file):
		"""
		This method takes an image and extract its descriptors,
		then index those descriptors in database with the image's product_id.

		Input
		------
		1- image_file from the flask request.
		2- product_id of the image.

		Output
		------
		returns is_success = True or False
		"""
		
		try:
			# extract descriptors.
			image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
			descriptors = cls.__extract_features(image)

			# generate unique name for the npy descriptors_file.
			descriptors_file_name = f"{uuid.uuid4().hex}.npy"

			# save descriptors file.
			file_path = cls.DESCRIPTORS_DIR + descriptors_file_name
			with open(file_path, 'wb') as f:
				np.save(f, descriptors)

			# save npy descriptors_file name and product_id in database.
			connection = sqlite3.connect(cls.DATABASE_PATH)
			cursor = connection.cursor()
			query = "INSERT INTO descriptors (name, image, product_id) VALUES (?, ?, ?)"
			cursor.execute(query, [descriptors_file_name, image_file.filename, product_id])
			connection.commit()
			connection.close()
			return True
		except Exception:
			return False








	@classmethod
	def predict(cls, query_image):
		"""
		This method takes a query image, extract its descriptor,
		then match it with descriptors found in the database.
		"""
		try:
			# extract query_image descriptors.
			image = cv2.imdecode(np.fromstring(query_image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
			query_image_descriptors = cls.__extract_features(image)

			# get all descriptors from database
			connection = sqlite3.connect(cls.DATABASE_PATH)
			cursor = connection.cursor()
			cursor.execute("SELECT name, product_id FROM descriptors")
			rows = cursor.fetchall()
			connection.close()

			# matching query_image descriptors with db_image descriptors
			all_ = []
			for name, product_id in rows:
				file_path = cls.DESCRIPTORS_DIR + name
				with open(file_path, 'rb') as f:
					db_image_descriptors = np.load(f)
					flann = cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 5), dict(checks=50))
					matches = flann.knnMatch(query_image_descriptors, db_image_descriptors, k=2)
					good_matches = [match1.distance for i,(match1,match2) in enumerate(matches) if match1.distance < (0.2 * match2.distance)]
					similarity_score = len(good_matches)
					result = {"product_id": product_id, "similarity_score": similarity_score}
					all_.append(result)
			all_ = sorted(all_, key=lambda x: x['similarity_score'], reverse=True)
			return all_[:4]
		except Exception:
			return False
    			



	@classmethod
	def __extract_features(cls, template_image):
		gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		_, descriptors = sift.detectAndCompute(gray,None)
		return descriptors
