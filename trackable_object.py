class TrackableObject:
	def __init__(self, objectID, centroid, class_id):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.class_id = class_id
		self.centroids = [centroid]
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False