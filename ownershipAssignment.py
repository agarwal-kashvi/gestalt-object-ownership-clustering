#Code for ownership assignment in GESTALT

#System Imports
import json
import warnings
import time
#Library Imports
import pandas as pd
from scipy.spatial import KDTree
import sklearn
from sklearn.cluster import KMeans
import Levenshtein
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from pyclustering.utils import calculate_distance_matrix
from pyclustering.cluster.birch import birch
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import numpy as np
import time
#User imports

class OwnershipAssigner():
	def __init__(self,locationData, objData):
		self._locationDict = locationData
		self._objectDict = objData 
		warnings.simplefilter("ignore") 									# Suppress warnings in pandas output


	def flatten_locations(self, locationsFile):
		'''
		Function to take the fully expressive Locations from openStreetMaps and squash into a flatter dict to be made into data frames
		Input Args: 
			locationsFile - dict of dicts - contains all the locations within the bounding box from OSM. 
		Operations: 
			- Iterate through the dictionary, generate lists
		Output
			- flatLocations - dict of lists
		'''

		flatLocations = {}															#Initialize vars
		locations = []
		latitudes = []
		longitudes = []

		for loc in locationsFile.keys():										#Loop through dict & append vals to list
			locations.append(loc)
			latitudes.append(locationsFile[loc]['latitude'])
			longitudes.append(locationsFile[loc]['longitude'])

		flatLocations["location"] = locations 										#Create the flattened dict of lists. 
		flatLocations["latitude"] = latitudes
		flatLocations["longitude"] = longitudes

		return flatLocations

	def flatten_objects_from_osm_dump(self, objectsDict):
		print("Starting to flatten_objects_from_osm_dump")

		flatObjects = {}
		flatObjects["object"] = []
		flatObjects["latitude"] = []
		flatObjects["longitude"] = []

		for object in objectsDict.keys():
			flatObjects["object"].append(objectsDict[object]['name'])
			flatObjects["latitude"].append(objectsDict[object]['latitude'])
			flatObjects["longitude"].append(objectsDict[object]['longitude'])

		return flatObjects
			



	def flatten_objects_from_kml(self, region):
		'''
		Function to take the fully expressive objects from the KML and squash into a flatter dict to be made into data frames
		Input Args: 
			region - string - the name of the region within the dict of objects to be flattened. 
			(implicit) self._objectDict - dict of dicts - contains all the objects within the bounding box from OSM. 
		Operations: 
			- Iterate through the dictionary, generate lists
		Output
			- flatOBJ - dict of lists
		'''

		flatOBJ = {} 															# Initialize vars
		locations = []
		objects = []
		latitudes = []
		longitudes = []

		attributeNumbers = []
		attributes = set() 														# Set used to generate list of unique descriptors

		for region in self._objDict.keys():										# Loop to get all the attribute descriptors. 
			for loc in self._objDict[region].keys():
				for obj in self._objDict[region][loc].keys():
					if self._objDict[region][loc][obj]["description"] != None:
						attributeNumbers.append(len(self._objDict[region][loc][obj]["description"]))
						for key in self._objDict[region][loc][obj]["description"].keys():
							attributes.add(list(self._objDict[region][loc][obj]["description"][key].keys())[0])
								
					else:
						attributeNumbers.append(0)


		flatOBJ["object"] = [] 											# Construct the dictionary 
		flatOBJ["latitude"] = []
		flatOBJ["longitude"] = []
		flatOBJ["true_location"] = []
		for attribute in attributes: 									# Add in keys and empty lists for each descriptor
			flatOBJ[attribute] = []



		for loc in self._objDict[region].keys():								# Loop through each object
			for obj in self._objDict[region][loc].keys():
				#print(self._objDict[region][loc][obj])
				flatOBJ["object"].append(self._objDict[region][loc][obj]['name'])
				flatOBJ["latitude"].append(self._objDict[region][loc][obj]['latitude'])
				flatOBJ["longitude"].append(self._objDict[region][loc][obj]['longitude'])
				flatOBJ["true_location"].append(loc)

				usedDescriptors = [] 											#Loop through each descriptor for an object, append to respective list or append None
				if self._objDict[region][loc][obj]['description'] is not None:
					for descriptor in self._objDict[region][loc][obj]['description'].keys():
						for key in self._objDict[region][loc][obj]['description'][descriptor].keys():
							#try:
							flatOBJ[key].append((self._objDict[region][loc][obj]['description'][descriptor][key]))
							#except KeyError:
							#	flatOBJ[key] = []
							#	flatOBJ[key].append((self._objDict[region][loc][obj]['description'][descriptor][key]))
							usedDescriptors.append(key)

				for attribute in attributes:
					if attribute not in usedDescriptors:
						#try:
						flatOBJ[attribute].append(None)
						usedDescriptors.append(attribute)
						#except KeyError:
						#	flatOBJ[attribute] = []
						#	flatOBJ[attribute].append(None)


		for obj in flatOBJ.keys():
			if len(flatOBJ[obj]) > len(flatOBJ["object"]): #Hacky workaround to get dataframes to be same length. TODO: Fix bug. 
				del flatOBJ[obj][-1]

		return flatOBJ

	def convertToDataFrame(self, flatLocations, flatObjects):								# Convert two flattened dictionaries into data frames

		#print("\n\n FLAT LOCATIONS:", flatLocations.items(),"\n\n")
		#print("\n\n FLAT OBJECTS:", flatObjects.items(),"\n\n")


		self._df_locations = pd.DataFrame.from_dict(flatLocations, orient="index")
		
		self._df_objects = pd.DataFrame.from_dict(flatObjects, orient="index")
		self._locationCoordinates = []
		self._locationIndex = {}

		i = 0

		#print("\n\n=====",self._df_locations,"=====\n\n",)
		for index, row in self._df_locations.iterrows():
			elem = [row[2],row[1]]								#Long, Lat
			self._locationCoordinates.append(elem)
			#print("LOCATION INDEX", row[0])
			#print("LOCATION COORDINATES", elem)

			self._locationIndex[i] = row[0]
			i+=1

		#print("\n\n=====",self._locationCoordinates,"=====\n\n",)
		self._location_kdTree = KDTree(self._locationCoordinates)

		self._objectCoordinates = []

		for index, row in self._df_objects.iterrows():
			elem = [row[2],row[1]]							#Long, lat
			self._objectCoordinates.append(elem)
			#self._objectIndex[index] = row[0]

		self._objects_kdTree = KDTree(self._objectCoordinates)

		print("Converted objects and OSM details to DataFrames")

		return ((self._df_locations, self._df_objects))

	def printToFile(self):
		'''
		Function to flatten coordiantes into a 0-100 grid for vizualization. 
		Input Args: 
			- boundingBox - list of floats - defines the max and min x and y coords to serve as 0 and 100. 
			- (implicit) self._df_osm - pandas dataframe containing the names, lat and longs of locations. 
			- (implicit) self._df_obj - pandas dataframe containing the names, lat, longs and parent locations of objects. 
		Actions: 
			- Use minimax normalization to make the bounding box go from 0:100
			- Append minimax normalized coordinates to end of dataframe
		Returns: 
			- Prints to csv the modified self._df_osm and self._df_obj dataframes. 
 		'''

 		#Print the dataframes to file. 

		self._df_locations.to_csv("data/osm_df.csv", index=False)
		self._df_objects.to_csv("data/obj_df.csv", index=False)

	def visualize_clusters(self, method_name, extra_info=""):
		"""
		Visualize clustering results
		
		Parameters:
		- method_name: Name of the clustering method
		- extra_info: Additional information to show in the title (e.g., parameters)
		"""
		plt.figure(figsize=(12, 10))
		
		# Plot objects colored by cluster
		scatter = plt.scatter(self._df_objects.longitude, self._df_objects.latitude, 
				   c=self._df_objects.cluster, cmap='tab20', alpha=0.6, s=50)
		
		# Plot locations as red stars
		plt.scatter(self._df_locations['longitude'], self._df_locations['latitude'], 
				   color='red', marker='*', s=100, label='Locations')
		
		# Add colorbar and labels
		plt.colorbar(scatter, label='Cluster')
		plt.xlabel('Longitude')
		plt.ylabel('Latitude')
		
		# Create title
		n_clusters = len(np.unique(self._df_objects['cluster']))
		n_noise = sum(self._df_objects['cluster'] == -1) if -1 in self._df_objects['cluster'].values else 0
		title = f'{method_name.upper()} Clustering Results\n{n_clusters} clusters'
		if n_noise > 0:
			title += f', {n_noise} noise points'
		if extra_info:
			title += f'\n{extra_info}'
		plt.title(title)
		
		plt.legend()
		plt.savefig(f'../data/output/{method_name.lower()}_clusters.png')
		plt.close()

	def plot_decision_boundary(self, method_name, model, X, filename, extra_info=""):
		"""
		Plot decision boundaries for clustering methods, only for data points with ground truth locations.
		
		Parameters:
		- method_name: Name of the clustering method (kmeans, dbscan, spectral, birch)
		- model: The fitted clustering model
		- X: Input data used for clustering (in [lat, lon] order)
		- filename: Output filename for the plot
		- extra_info: Additional information to show in the title
		"""
		plt.figure(figsize=(12, 10))
		
		# Debug prints
		print(f"\nDebug info for {method_name} decision boundary plot:")
		print(f"Total number of objects: {len(self._df_objects)}")
		print(f"Number of objects with ground truth: {self._df_objects['ground_truth_location'].notna().sum()}")
		print(f"Shape of input data X: {X.shape}")
		
		# Filter data points with ground truth locations
		has_ground_truth = self._df_objects['ground_truth_location'].notna()
		X_filtered = X[has_ground_truth]
		clusters_filtered = self._df_objects.loc[has_ground_truth, 'cluster']
		
		print(f"Shape of filtered data: {X_filtered.shape}")
		print(f"Number of unique clusters: {len(np.unique(clusters_filtered))}")
		
		if len(X_filtered) == 0:
			print("Warning: No data points with ground truth found!")
			plt.close()
			return
		
		# Create a mesh grid only for the region with ground truth data
		x_min, x_max = X_filtered[:, 0].min() - 0.1, X_filtered[:, 0].max() + 0.1
		y_min, y_max = X_filtered[:, 1].min() - 0.1, X_filtered[:, 1].max() + 0.1
		
		print(f"Mesh grid bounds: x[{x_min:.4f}, {x_max:.4f}], y[{y_min:.4f}, {y_max:.4f}]")
		
		xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
							np.arange(y_min, y_max, 0.01))
		
		# Get predictions for mesh grid points
		mesh_points = np.c_[xx.ravel(), yy.ravel()]
		print(f"Number of mesh points: {len(mesh_points)}")
		
		if method_name == "kmeans":
			Z = model.predict(mesh_points)
		elif method_name == "dbscan":
			Z = model.fit_predict(mesh_points)
		elif method_name == "spectral":
			Z = model.fit_predict(mesh_points)
		elif method_name == "birch":
			Z = model.predict(mesh_points)
		
		# Reshape predictions to match mesh grid
		Z = Z.reshape(xx.shape)
		
		# Plot decision boundaries
		plt.contourf(xx, yy, Z, alpha=0.4, cmap='tab20')
		
		# Plot original data points with ground truth
		scatter = plt.scatter(X_filtered[:, 1], X_filtered[:, 0],  # Swap lat/lon for plotting
							c=clusters_filtered,
							cmap='tab20', alpha=0.6, s=50)
		
		# Plot locations as red stars
		plt.scatter(self._df_locations['longitude'], self._df_locations['latitude'],
				   color='red', marker='*', s=100, label='Locations')
		
		# Add colorbar and labels
		plt.colorbar(scatter, label='Cluster')
		plt.xlabel('Longitude')
		plt.ylabel('Latitude')
		
		# Create title
		n_clusters = len(np.unique(clusters_filtered))
		n_noise = sum(clusters_filtered == -1) if -1 in clusters_filtered.values else 0
		title = f'{method_name.upper()} Decision Boundaries\n{n_clusters} clusters'
		if n_noise > 0:
			title += f', {n_noise} noise points'
		if extra_info:
			title += f'\n{extra_info}'
		plt.title(title)
		
		plt.legend()
		plt.savefig(f'../data/SV/output/ownershipAssignment/plot/{filename}')
		plt.close()

	def fuzzy_multiple_assignment(self, centroids, centroid_labels, fuzzy_threshold):
		print("Fuzzy multiple assignment")
		df_multi_asn_objects = self._df_objects.copy()
		for idx, row in self._df_objects.iterrows():
			for c_i, centroid in enumerate(centroids):
				if ((centroid_labels[c_i] != -1) and centroid_labels[c_i] != row['cluster']): # dont care to add to NULL as we dont search it nor same cluster already in as that causes dups
					obj_centroid_dist = self.__distance__((row['latitude'], row['longitude']), centroid)
					# if obj-centroid distance is within THRESHOLD% of range of obj-centroid distances we saw during exact assignment  
					threshold = fuzzy_threshold * (self.cluster_max_dist - self.cluster_min_dist)
					if (obj_centroid_dist - self.cluster_min_dist) < threshold:
							row['cluster'] = centroid_labels[c_i]
							row_df = pd.DataFrame([row], columns=df_multi_asn_objects.columns)
							df_multi_asn_objects = pd.concat([df_multi_asn_objects, row_df], ignore_index=True) #df_multi_asn_objects.append(row)
		return df_multi_asn_objects
	
	def calculate_distances(self, centroids):
		dists = []
		for idx, row in self._df_objects.iterrows():
			obj_coord = np.array([row['latitude'], row['longitude']])
			centroid_coord = centroids[row['cluster']]
			dists.append(self.__distance__(obj_coord, centroid_coord))
		return dists

	def kMeans_membership(self, numberOfClusters, fuzzy_threshold=0):
		print("Clustering with kMeans")
		
		kmeans_cluster = KMeans(n_clusters=numberOfClusters, random_state=0, n_init="auto")
		
		# Ensure we're using [lat, lon] order consistently
		X = self._df_objects[['latitude', 'longitude']].values
		
		start_wall_time = time.time()
		start_proc_time=time.process_time()
		self._df_objects['cluster'] = kmeans_cluster.fit_predict(X)
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO CLUSTER OBJECTS:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO CLUSTER OBJECTS:", end_wall_time-start_wall_time,"\n")
		
		centroids = kmeans_cluster.cluster_centers_
		centroid_labels = kmeans_cluster.labels_
		assert all(centroid_labels == self._df_objects['cluster']), "Cluster labels do not match"

		# Calculate distances and probabilities
		dists = self.calculate_distances(centroids)
		self._df_objects['assignment_prob'] = 1 - self.__normalize_probs__(dists, mask=list(self._df_objects['cluster'] != -1))

		# Plot decision boundaries
		X = self._df_objects[['latitude', 'longitude']].values
		self.plot_decision_boundary("kmeans", kmeans_cluster, X, "kmeans_decision_boundary.png", 
								  f"n_clusters={numberOfClusters}")

		# Fuzzy multiple assignment
		if fuzzy_threshold > 0:
			self._df_objects = self.fuzzy_multiple_assignment(centroids, centroid_labels, fuzzy_threshold)
		
		self.inferLocation(self._df_objects, centroids, "kmeans")
	
	def dbscan_membership(self, epsilon=0.5/6371., minCluster=3, fuzzy_threshold=0):
		"""
		- epsilon=0.5/6371 means ~50 meters (0.5 km / Earth's radius in km)
		- Using np.radians() converts coordinates to radians, making distances meaningful in terms of Earth's radius
		- No scaling needed because epsilon is already in terms of Earth's radius
		"""
		print("Clustering with DBScan")
		loc_arr = np.array(self._objectCoordinates)

		start_wall_time = time.time()
		start_proc_time=time.process_time()
		db_cluster = DBSCAN(eps=epsilon, min_samples=minCluster).fit(np.radians(loc_arr))
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO CLUSTER OBJECTS:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO CLUSTER OBJECTS:", end_wall_time-start_wall_time,"\n")
		
		self._df_objects['cluster'] = db_cluster.labels_

		centroids = self.calculateCentroids(db_cluster.labels_)
		centroid_labels = db_cluster.labels_

		dists = self.calculate_distances(centroids)
		
		self._df_objects['assignment_prob'] = 1 - self.__normalize_probs__(dists, mask=list(self._df_objects['cluster'] != -1))

		# Plot decision boundaries
		X = np.radians(loc_arr)  # Use radians for DBSCAN
		self.plot_decision_boundary("dbscan", db_cluster, X, "dbscan_decision_boundary.png",
								  f"eps={epsilon:.2e}, min_samples={minCluster}")
  
		# Fuzzy multiple asn
		if fuzzy_threshold > 0:
			self._df_objects = self.fuzzy_multiple_assignment(centroids, centroid_labels, fuzzy_threshold)

		self.inferLocation(self._df_objects, centroids, "dbscan")

	def spectral_membership(self, n_clusters, gamma=1.0, fuzzy_threshold=0):
		"""
		- n_clusters: number of clusters (should match number of locations)
		- gamma: kernel coefficient for RBF kernel, controls how fast similarity decreases with distance
		         smaller gamma = slower decrease = larger clusters
		"""
		print("Clustering with SpectralClustering")
		print(f"Parameters: n_clusters={n_clusters}, gamma={gamma}")
		
		# Convert coordinates to numpy array and ensure [lat, lon] order for consistency
		loc_arr = np.array([[coord[1], coord[0]] for coord in self._objectCoordinates])
		print(f"Number of objects to cluster: {len(loc_arr)}")
		
		# Scale the features to have zero mean and unit variance
		scaler = StandardScaler()
		loc_arr_scaled = scaler.fit_transform(loc_arr)
		
		# Initialize and run SpectralClustering
		spectral = SpectralClustering(
			n_clusters=n_clusters,
			gamma=gamma,
			random_state=42,
			n_jobs=-1  # Use all CPU cores
		)
		
		# Get cluster labels
		start_wall_time = time.time()
		start_proc_time=time.process_time()
		self._df_objects['cluster'] = spectral.fit_predict(loc_arr_scaled)
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO CLUSTER OBJECTS:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO CLUSTER OBJECTS:", end_wall_time-start_wall_time,"\n")

		unique_clusters = np.unique(self._df_objects['cluster'])
		cluster_sizes = [sum(self._df_objects['cluster'] == i) for i in unique_clusters]
		print(f"Unique cluster labels: {unique_clusters}")
		print(f"Cluster sizes: {cluster_sizes}")
		print(f"Average cluster size: {np.mean(cluster_sizes):.1f}")
		print(f"Median cluster size: {np.median(cluster_sizes):.1f}")
		
		# Calculate cluster centroids
		centroids = []
		for cluster_id in range(n_clusters):
			cluster_points = loc_arr[self._df_objects['cluster'] == cluster_id]
			if len(cluster_points) > 0:
				centroid = np.mean(cluster_points, axis=0)
				centroids.append(centroid)
		centroids = np.array(centroids)
		
		print(f"Number of centroids: {len(centroids)}")
		
		# Calculate distances and probabilities
		dists = self.calculate_distances(centroids)
		self._df_objects['assignment_prob'] = 1 - self.__normalize_probs__(dists, mask=list(self._df_objects['cluster'] != -1))
		
		# Plot decision boundaries
		self.plot_decision_boundary("spectral", spectral, loc_arr_scaled, "spectral_decision_boundary.png",
								  f"n_clusters={n_clusters}, gamma={gamma}")
		
		# Convert centroids back to [lon, lat] order for inferLocation
		centroids_converted = [[c[1], c[0]] for c in centroids]

		# Infer locations
		self.inferLocation(self._df_objects, centroids_converted, "spectral")


	def birch_membership(self, n_clusters, threshold, branching_factor, fuzzy_threshold=0):
		"""
		Clustering with BIRCH
		- threshold is in terms of Euclidean distance
		- Using scale_factor=111 (km/degree) to convert coordinate differences to meaningful distances
		- Example: threshold=0.5 with scale_factor=111 means ~55.5 km threshold
		- Without scaling, raw coordinate differences (e.g., 0.0001 degrees) would be too small for effective clustering
		"""
		print("Clustering with BIRCH")
		print(f"Parameters: n_clusters={n_clusters}, threshold={threshold}, branching_factor={branching_factor}")
		birch = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)

		# Convert coordinates to numpy array and ensure [lat, lon] order for consistency
		loc_arr = np.array([[coord[1], coord[0]] for coord in self._objectCoordinates])
		print(f"Number of objects to cluster: {len(loc_arr)}")
		
		# Scale coordinates to help with clustering. Convert to kilometers (approximately)
		scale_factor = 111  # 1 degree ≈ 111 kilometers
		loc_arr_scaled = loc_arr * scale_factor
		
		# Get cluster labels
		start_wall_time = time.time()
		start_proc_time=time.process_time()
		self._df_objects['cluster'] = birch.fit_predict(loc_arr_scaled)
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO CLUSTER OBJECTS:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO CLUSTER OBJECTS:", end_wall_time-start_wall_time,"\n")
		
		unique_clusters = np.unique(self._df_objects['cluster'])
		print(f"Unique cluster labels: {unique_clusters}")
		print(f"Cluster sizes: {[sum(self._df_objects['cluster'] == i) for i in unique_clusters]}")
		
		# Calculate final cluster centroids
		#centroids = {}
		#for cluster_id in range(n_clusters):
			#cluster_points = loc_arr[self._df_objects['cluster'] == cluster_id]
			#if len(cluster_points) > 0:
			#	centroid = np.mean(cluster_points, axis=0)
			#	#centroids.append(centroid)
		#centroids = np.array(centroids)
		centroids = {}
		clusters_np = self._df_objects["cluster"].to_numpy()
		for cluster_id in unique_clusters:
			cluster_points = loc_arr[clusters_np == cluster_id]
			if cluster_points.shape[0] > 0:
				centroids[cluster_id] = np.mean(cluster_points, axis=0)
		print(f"Number of final centroids: {len(centroids)}")
		
		# Calculate distances and probabilities
		#dists = self.calculate_distances(centroids)
		dists = []
		for idx, row in self._df_objects.iterrows():
			cluster_id = row["cluster"]
			obj_coord = np.array([row["latitude"], row["longitude"]])
			centroid_coord = centroids[cluster_id]
			dists.append(self.__distance__(obj_coord, centroid_coord))
		self._df_objects['assignment_prob'] = 1 - self.__normalize_probs__(dists, mask=list(self._df_objects['cluster'] != -1))
		
		# Plot decision boundaries
		#self.plot_decision_boundary("birch", birch, loc_arr_scaled, "birch_decision_boundary.png",
								 # f"n_clusters={n_clusters}, threshold={threshold}, branching_factor={branching_factor}")
		
		# Convert centroids back to [lon, lat] order for inferLocation
		#centroids_converted = [[c[1], c[0]] for c in centroids]
		
		# Infer locations
		#self.inferLocation(self._df_objects, centroids_converted, "birch")

		mappings = {}
		for cluster_id, centroid in centroids.items():
			centroid_lon_lat = [centroid[1], centroid[0]]
			d, i = self._location_kdTree.query(centroid_lon_lat, 1)
			mappings[cluster_id] = self._locationIndex[i]
		self._df_objects["predicted_location"] = self._df_objects["cluster"].map(mappings)

	def __distance__(self, point1, point2):
		return np.linalg.norm(point1 - point2)
  
	def __normalize_probs__(self, column, mask):
		# expects a list mask of booleans, where True means we account for the datapoint as valid max or min
		valid_data =  np.array(column)[np.array(mask)]

		# Set class vars for fuzzy multiple assignment if applicable    
		self.cluster_min_dist = np.min(valid_data)  
		self.cluster_max_dist = np.max(valid_data)
  
		return_col = np.array((column - np.min(valid_data)) / (np.max(valid_data) - np.min(valid_data)))
		return_col[~np.array(mask)] = 0.5  # forcing the ones we don't count to have prob = 0.5
		return return_col

	def calculateCentroids(self, clusters):
		print("Calculating Centroids")
		centroids = []

		for cluster in range (0, (max(clusters)+1)): 								#+1 to account for indexing from 0
			cluster_df = self._df_objects.loc[self._df_objects['cluster'] == cluster]		# Get only the coords belonging to this cluster
			coords = []

			for index, obj in cluster_df.iterrows(): 								# Make the coords into a list, then numpy array
				coords.append([obj.latitude, obj.longitude])
			np_coords = np.array(coords)
			
			centroid = np.mean(np_coords,axis=0) 									# Get the midpoint of the array
			centroids.append(centroid) 												# Build list of centroids

		return(centroids)


	def inferLocation(self, objs_to_assign_df, centroids, method):
		#print("Inferring object location")
		mappings = {} 																#Dict so that arbitrary number of clusters can be used
		for centroid in range (0, (len(centroids))): 								# For each centroid
			d, i = self._location_kdTree.query(centroids[centroid],1) 				# Look up its nearest neighbour in the KD tree
			#idx = (list(self._locationIndex.keys()))[i]
			#print(self._locationIndex[idx])
			#print(self._locationIndex[i])
			#print(centroids[centroid])
			mappings[centroid] = self._locationIndex[i]

		objs_to_assign_df['predicted_location'] = objs_to_assign_df.cluster.map(mappings) 		# Infer that the nearest neighbour is the cluster location
	

	def evaluateClusters(self, df_to_eval, method):
		#Move this to own function later. Use Levenshtein at 0.7 to handle labelling differences. 
		matches = []

		for index,row in self._df_objects.iterrows():								
			if Levenshtein.ratio(row['predicted_location'], row["true_location"]) >= 0.7:
				matches.append("True")
			else:
				matches.append("False")

		df_to_eval[method+"_correct"] = matches

		print(df_to_eval)
	
	def seeded_iterative_membership(self, n_nearest=None, noise_scale=2.0, max_iters=50, fuzzy_threshold=0):
		print("Running seeded iterative clustering with noise culling")
		start_wall_time = time.time()
		start_proc_time = time.process_time()
		obj_coords = self._df_objects[['latitude', 'longitude']].to_numpy()
		loc_coords = self._df_locations[['latitude', 'longitude']].to_numpy()
		num_objects = len(obj_coords)
		num_locations = len(loc_coords)
		if n_nearest is None:
			if num_locations > 0:
				n_nearest = max(1, round(num_objects / num_locations))
			else:
				n_nearest = 1
		print("num_objects =", num_objects)
		print("num_locations =", num_locations)
		print("n_nearest=", n_nearest)
		if num_objects == 0 or num_locations == 0:
			print("No objects or locations available for clustering.")
			self._df_objects['is_noise'] = True
			self._df_objects['cluster'] = -1
			self._df_objects['assignment_prob'] = 0.0
			return
		# Keep original location IDs persistent
		cluster_ids = np.arange(num_locations)
		# nearest location for each object
		t0 = time.time()
		print("Building KD-tree for locations...")
		loc_tree = cKDTree(loc_coords)
		print("Querying nearest location for each object...")
		nearest_loc_dist, nearest_loc_idx = loc_tree.query(obj_coords, k=1)
		print("Done object -> nearest location in", time.time() - t0, "seconds")
		# nearest N objects for each location
		t1 = time.time()
		print("Building KD-tree for objects...")
		obj_tree = cKDTree(obj_coords)
		k_for_locations = min(max(1, n_nearest), num_objects)
		print("Querying nearest", k_for_locations, "objects for each location...")
		loc_to_obj_dists, loc_to_obj_idx = obj_tree.query(loc_coords, k=k_for_locations)
		if k_for_locations == 1:
			loc_to_obj_dists = loc_to_obj_dists[:, np.newaxis]
		nearest_n_dists = loc_to_obj_dists.ravel()
		print("Done location -> nearest N objects in", time.time() - t1, "seconds")
		# noise threshold
		if len(nearest_n_dists) > 0:
			base = np.median(nearest_n_dists)
			mad = np.median(np.abs(nearest_n_dists - base))
			if mad == 0:
				mad = np.std(nearest_n_dists)
			noise_threshold = base + noise_scale * mad
		else:
			noise_threshold = np.inf
		print("noise_threshold =", noise_threshold)
		is_noise = nearest_loc_dist > noise_threshold
		self._df_objects['is_noise'] = is_noise
		valid_mask = ~is_noise
		valid_obj_coords = obj_coords[valid_mask]
		print("valid objects =", len(valid_obj_coords))
		print("noise objects =", np.sum(is_noise))
		if len(valid_obj_coords) == 0:
			print("All objects were marked as noise. No clustering performed.")
			self._df_objects['cluster'] = -1
			self._df_objects['assignment_prob'] = 0.0
			return
		centroids = loc_coords.copy()
		prev_assignments = None
		final_assignments = None
		for it in range(max_iters):
			iter_start = time.time()
			centroid_tree = cKDTree(centroids)
			dists, assignments = centroid_tree.query(valid_obj_coords, k=1)
			print("iteration", it, "- assigned in", time.time() - iter_start, "seconds")
			if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
				final_assignments = assignments
				print("Converged at iteration", it)
				break
			new_centroids = centroids.copy()
			for cid in range(num_locations):
				cluster_points = valid_obj_coords[assignments == cid]
				if len(cluster_points) > 0:
					new_centroids[cid] = np.mean(cluster_points, axis=0)
			centroids = new_centroids
			prev_assignments = assignments
			final_assignments = assignments
		if final_assignments is None:
			centroid_tree = cKDTree(centroids)
			_, final_assignments = centroid_tree.query(valid_obj_coords, k=1)
		full_cluster_labels = np.full(num_objects, -1, dtype=int)
		full_cluster_labels[valid_mask] = cluster_ids[final_assignments]
		self._df_objects['cluster'] = full_cluster_labels
		centroid_tree = cKDTree(centroids)
		min_valid_dists, _ = centroid_tree.query(valid_obj_coords, k=1)
		if len(min_valid_dists) > 1 and np.max(min_valid_dists) != np.min(min_valid_dists):
			conf = 1 - (
            (min_valid_dists - np.min(min_valid_dists)) /
            (np.max(min_valid_dists) - np.min(min_valid_dists))
        	)
		else:
			conf = np.ones(len(min_valid_dists))
		full_conf = np.zeros(num_objects)
		full_conf[valid_mask] = conf
		self._df_objects['assignment_prob'] = full_conf
		end_proc_time = time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO CLUSTER OBJECTS:", end_proc_time - start_proc_time)
		print("WALL TIME TAKEN TO CLUSTER OBJECTS:", end_wall_time - start_wall_time, "\n")
		if fuzzy_threshold > 0:
			self._df_objects = self.fuzzy_multiple_assignment(centroids, cluster_ids, fuzzy_threshold)
		self.inferLocation(self._df_objects, centroids, "seeded_iterative")
