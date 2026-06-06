#System Imports
import argparse, json, sys, os

import time

#Library Imports
import pickle

#User Imports

from dataCollection import TerrainExtractor, osmQueryEngine, PhotoDownloader
from ownershipAssignment import OwnershipAssigner
from conceptMapping import ConceptMapper
from search import InvertedIndex



if __name__ == "__main__":

	argparser = argparse.ArgumentParser()									# initialize the argParser

	argparser.add_argument(	"-ql", "--queryOsmLocations", 							
							help="engage mode to issue a query to OpenStreetMaps for locations",
							action="store_true",
							default=False,
							required=False)	

	argparser.add_argument(	"-qo", "--queryOsmObjects", 							
							help="engage mode to issue a query to OpenStreetMaps for objects",
							action="store_true",
							default=False,
							required=False)	
		
	argparser.add_argument( "-b", "--boundingbox", 
							help="specifies the bounding box to be passed to OSM", 
							nargs='+',
							default=None,
							required=False)
	
	argparser.add_argument( "-p", "--photosFromFlickr",
							help="engage mode to ingest the processed metadata_objects.json from the PhotoDownloader. Needs to be used in conjunction with -f and -o",
							action="store_true",
							default=False,
							required=False)
	
	argparser.add_argument( "-s", "--searchterms",
							help="search terms to pass to the overpass turbo API",
							nargs="+",
							default=None,
							required=False)
	
	argparser.add_argument( '-o', "--output",
							help="directory to output results to", 
							type=str,
							default="data/", 
							required=False)
	
	argparser.add_argument(	"-k", "--kmlIngestMode", 							
							help="Mode to ingest KML files",
							action="store_true",
							default=False,
							required=False)	
	
	argparser.add_argument( "-f", "--fileSource",
							help="kmlfile(s) to ingest)", 
							nargs="+", 
							default=None,
							required=False)

	argparser.add_argument(	"-oa", "--ownershipAssignment", 							
							help="Define the location membership inference method: 'kmeans', 'dbscan', 'birch', 'spectral', 'seeded_iterative'",
							type=str,
							default="None",
							required=False)

	argparser.add_argument("-id", "--inputDirectory",
							help="The directory to get files from for the Ownership Assignment", 
							type=str, 
							required=False)
	
	argparser.add_argument("-od", "--outputDirectory",
							help="The directory to write files created by the Ownership Assignment", 
							type=str, 
							required=False)
	
	argparser.add_argument("-n", "--numClusters", 
							help="number of clusters to use for KMeans",
							type=int, 
							required=False)
	
	argparser.add_argument("-e", "--epsilon", 
							help="Epsilon value for DBSCAN - Float",
							type=float, 
							required=False)
	
	argparser.add_argument("-bt", "--birchThreshold",
							help="Threshold value for BIRCH clustering",
							type=float,
							default=0.25,
							required=False)
	
	argparser.add_argument("-bf", "--birchBranchingFactor",
							help="Branching factor for BIRCH clustering",
							type=int,
							default=150,
							required=False)
	
	argparser.add_argument("-gs", "--gestaltSearch", 
							help="Invoke Gestalt's Search Mode",
							action="store_true",
							required=False)
	
	argparser.add_argument("-if", "--inputFile", 
							help="Input file to create the inverted index from",
							type=str,
							required=False)
	
	argparser.add_argument(	"-pd", "--photoDownloader", 							
							help="Mode to download files from flickr",
							action="store_true",
							default=False,
							required=False)	
	
	argparser.add_argument(	"-ccm", "--createConceptMaps", 							
							help="Create concept maps based on the predicted locations",
							action="store_true",
							default=False,
							required=False)	
	
	argparser.add_argument(	"-ft", "--fuzzy_threshold", 							
							help="Threshold Value for Fuzzy Search",
							type=float,
							default=0.0,
							required=False)	
	
	argparser.add_argument(	"-fpn", "--flickrPageNumber", 							
							help="PageNumber to Start processing flickr results from",
							type=int,
							default=1,
							required=False)	
	
	argparser.add_argument(	"-ii", "--dumpInvertedIndex", 							
							help="Dumps the Inverted Index",
							action="store_true",
							default=False,
							required=False)		
	
	argparser.add_argument("-gamma", "--spectralGamma",
							help="Gamma parameter for Spectral clustering (controls cluster size)",
							type=float,
							default=15.0,
							required=False)
	
	argparser.add_argument("-nn", "--n_nearest", 
						type=int, 
						default=None, 
						help="Number of nearest points for seeded_iterative. Default = objects/locations ratio"
)

	flags = argparser.parse_args()																		# populate variables from command line arguments


	if flags.queryOsmObjects== True: 																		# Check to see if GESTALT should execute in OSM Query mode. 
		
		start_wall_time = time.time()
		start_proc_time=time.process_time()
		
		bbox = flags.boundingbox
		outputfile = flags.output
		searchterms = flags.searchterms
		oqe = osmQueryEngine()
		
		if searchterms[0] == "allobjects":
			osmDict, vocab = oqe.queryOSMforObjects(bbox)
			outputfileName = (outputfile+"_"+"".join(bbox)+"_allobjects.json")
		else:
			osmDict = oqe.queryOSMforObjects(bbox, searchterms[0]) 													# Create the dictionary off the first search term 
			for i in range(1, len(searchterms), 1): 														# Loop through other search terms and merge each set of results into one big dict
				newDict = oqe.queryOSMforObjects(bbox, searchterms[i])
				osmDict.update(newDict)
			outputfileName = (outputfile+"".join(searchterms)+".json") 												#Define the filename to output to
		
		with open(outputfileName,'w') as outfile: 															#dump output to json
			json.dump(osmDict, outfile, indent=4)
			print("Successfully outputted data to \"{outputLoc}\"".format(outputLoc=outputfileName)) 		#user feedback. 
		
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO QUERY OSM OBJECTS:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO QUERY OSM OBJECTS:", end_wall_time-start_wall_time,"\n")
		exit()

	if flags.queryOsmLocations== True: 																		# Check to see if GESTALT should execute in OSM Query mode. 
		start_wall_time = time.time()
		start_proc_time=time.process_time()		
		
		bbox = flags.boundingbox
		outputfile = flags.output
		searchterms = flags.searchterms
		oqe = osmQueryEngine()
		
		if searchterms[0] == "alllocations":
			osmDict = oqe.queryOSMforLocations(bbox, searchterms[0])
			outputfileName = (outputfile+"_"+"".join(bbox)+"_alllocations.json")
		else:
			osmDict = oqe.queryOSMforLocations(bbox, searchterms[0]) 													# Create the dictionary off the first search term 
			for i in range(1, len(searchterms), 1): 														# Loop through other search terms and merge each set of results into one big dict
				newDict = oqe.queryOSMforLocations(bbox, searchterms[i])
				osmDict.update(newDict)
			outputfileName = (outputfile+"".join(bbox)+"_alllocations.json") 												#Define the filename to output to
		
		with open(outputfileName,'w') as outfile: 															#dump output to json
			json.dump(osmDict, outfile, indent=4)
			print("Successfully outputted data to \"{outputLoc}\"".format(outputLoc=outputfileName)) 		#user feedback. 

		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO QUERY OSM LOCATIONS:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO QUERY OSM LOCATIONS:", end_wall_time-start_wall_time,"\n")
		exit()

	if flags.kmlIngestMode == True: 																	# Check to see if gestalt should execute in KML Ingestion mode. 
		start_wall_time = time.time()
		start_proc_time=time.process_time()
		
		sourceFiles = flags.fileSource
		outputfile = flags.output
		tex = TerrainExtractor()
		for file in sourceFiles:
			print(file)
			fullfilename = file.split("/")[-1] 															#get everything right of final "/" (i.e. filename)
			shortFileName = fullfilename.split(".")[0] 													#get everything left of the "." (i.e not filetype)
			objectLocations, vocab = tex.Ingest_kml_file(file) 												#get the dictionary from ingesting KML file
			outputfileName = outputfile+"_"+shortFileName+".json" 										#set the output name to be distinct for the file
			with open(outputfileName,'w') as outfile: 													#output to JSON
				json.dump(objectLocations, outfile, indent=4)
			print("Successfully outputted data to \"{outputLoc}\"".format(outputLoc=outputfileName)) 	#user feedback
		
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO INGEST KML:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO INGEST KML:", end_wall_time-start_wall_time,"\n")		
		exit()

	if flags.photosFromFlickr==True:
		print("INGESTING FROM FLICKR")
		start_wall_time = time.time()
		start_proc_time=time.process_time()

		object_file = flags.inputFile
		outputfile = flags.output
		print("OUTPUT TO", outputfile)
		tex = TerrainExtractor()
		fullfilename = object_file.split("/")[-1] 													#get everything right of final "/" (i.e. filename)
		shortFileName = fullfilename.split(".")[0] 													#get everything left of the "." (i.e not filetype)
		vocab, objectLocations = tex.ingest_flickr_objects(object_file) 									#get the dictionary from ingesting KML file
		outputfileName = outputfile+"_"+shortFileName+".json" 										#set the output name to be distinct for the file
		with open(outputfileName,'w') as outfile: 													#output to JSON
			json.dump(objectLocations, outfile, indent=4)
		print("Successfully outputted data to \"{outputLoc}\"".format(outputLoc=outputfileName)) 	#user feedback
		
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO QUERY FLICKR:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO QUERY FLICKR:", end_wall_time-start_wall_time,"\n")
		exit()


	if flags.ownershipAssignment.lower() == "kmeans":
		prefix = flags.inputDirectory																#Read parameters for the file
		outputFile = flags.outputDirectory
		numClusters = flags.numClusters
		fuzzy_threshold = flags.fuzzy_threshold

		objectsDict = {}																			#Initilaize the Dicts
		locationsDict = {}
																									
		for file in os.listdir(prefix):																# Load in the files
			print("Adding",file,"to K-Means")
			
			if file.startswith("objects"): 															# Get the objects files
				with open(prefix+"/"+file, "r") as inObjs:
					objects = json.load(inObjs)
				objectsDict.update(objects)
			
			if file.startswith("locations"):														# Get the locations files
				with open(prefix+"/"+file, "r") as inLocs:
					locations = json.load(inLocs)
				locationsDict.update(locations)
		
		ownerAssigner = OwnershipAssigner(locations, objects) 										# Initalize the ownership assigner
		
		df_locations, df_objects = ownerAssigner.convertToDataFrame(locationsDict, objectsDict)		# Convert the dictionaries created from the JSON inputs to Pandas Dataframes

		ownerAssigner._df_locations.to_csv(outputFile+"/locations.csv", index=False) 				# Write the dataframes to file before clustering
		ownerAssigner._df_objects.to_csv(outputFile+"/objects.csv", index=False)

		ownerAssigner.kMeans_membership(len(locationsDict.keys()), fuzzy_threshold=fuzzy_threshold) 						# Conduct clustering

		clusters = ownerAssigner._df_objects["cluster"] 											# Infer Ownership
		print(clusters.value_counts())

		ownerAssigner._df_objects.to_csv(outputFile+f"/KMEANS_PredictedLocations_FT={fuzzy_threshold}.csv", index=False)	# Print the Clustered file
		
		exit()

	if flags.ownershipAssignment.lower() == "birch":
		prefix = flags.inputDirectory
		outputFile = flags.outputDirectory
		numClusters = flags.numClusters
		fuzzy_threshold = flags.fuzzy_threshold
		threshold = flags.birchThreshold
		branching_factor = flags.birchBranchingFactor

		objectsDict = {}
		locationsDict = {}

		print("\n\n- - - - - Ingesting Data to BIRCH - - - - - \n")
		for file in os.listdir(prefix):
			print("Adding",file,"to BIRCH", f"File path: {prefix}/{file}")
			
			if file.startswith("objects"):
				with open(prefix+"/"+file, "r") as inObjs:
					objects = json.load(inObjs)
				objectsDict.update(objects)
			
			if file.startswith("locations"):
				with open(prefix+"/"+file, "r") as inLocs:
					locations = json.load(inLocs)
				locationsDict.update(locations)
		print("DONE Adding data to BIRCH", f"len(locationsDict): {len(locationsDict)}", f"len(objectsDict): {len(objectsDict)}")
		
		ownerAssigner = OwnershipAssigner(locationsDict, objectsDict)
		df_locations, df_objects = ownerAssigner.convertToDataFrame(locationsDict, objectsDict)

		ownerAssigner._df_locations.to_csv(outputFile+"/locations.csv", index=False)
		ownerAssigner._df_objects.to_csv(outputFile+"/objects.csv", index=False)

		print("RUNNING BIRCH WITH N_CLUSTERS:", len(locationsDict.keys()), "THRESHOLD:", threshold, "BRANCHING_FACTOR:", branching_factor)
		
		ownerAssigner.birch_membership(n_clusters=len(locationsDict.keys()), threshold=threshold, branching_factor=branching_factor, fuzzy_threshold=fuzzy_threshold)

		ownerAssigner._df_objects.to_csv(outputFile+f"/BIRCH_PredictedLocations_N={len(locationsDict.keys())}_T={threshold}_BF={branching_factor}.csv", index=False)
				
		exit()

	if flags.ownershipAssignment.lower() == "dbscan":
		prefix = flags.inputDirectory																#Read parameters for the file
		outputFile = flags.outputDirectory
		numClusters = flags.numClusters
		epsilon= flags.epsilon
		minCluster = flags.numClusters
		fuzzy_threshold = flags.fuzzy_threshold

		objectsDict = {}																			#Initilaize the Dicts
		locationsDict = {}

		print("\n\n- - - - - Ingesting Data to DBSCAN - - - - - \n")
		for file in os.listdir(prefix):																# Load in the files
			print("Adding",file,"to DBSCAN", f"File path: {prefix}/{file}")
			
			if file.startswith("objects"): 															# Get the objects files
				with open(prefix+"/"+file, "r") as inObjs:
					objects = json.load(inObjs)
				objectsDict.update(objects)
			
			if file.startswith("locations"):														# Get the locations files
				with open(prefix+"/"+file, "r") as inLocs:
					locations = json.load(inLocs)
				locationsDict.update(locations)
		print("DONE Adding data to DBSCAN", f"len(locationsDict): {len(locationsDict)}", f"len(objectsDict): {len(objectsDict)}")
		ownerAssigner = OwnershipAssigner(locationsDict, objectsDict) 										# Initalize the ownership assigner
		
		df_locations, df_objects = ownerAssigner.convertToDataFrame(locationsDict, objectsDict)		# Convert the dictionaries created from the JSON inputs to Pandas Dataframes

		ownerAssigner._df_locations.to_csv(outputFile+"/locations.csv", index=False) 				# Write the dataframes to file before clustering
		ownerAssigner._df_objects.to_csv(outputFile+"/objects.csv", index=False)

		print("RUNNING DBSCAN WITH EPSILON:", epsilon, "MINCLUSTER:", minCluster, "FUZZY_THRESHOLD", fuzzy_threshold)
		
		ownerAssigner.dbscan_membership(epsilon,minCluster,fuzzy_threshold=fuzzy_threshold)	# Cluster the objects
		
		clusters = ownerAssigner._df_objects["cluster"] 											# Infer the location
		#print(clusters.value_counts())

		ownerAssigner._df_objects.to_csv(outputFile+"/DBSCAN_PredictedLocations_FT="+str(fuzzy_threshold)+".csv", index=False)	# Save to file

		exit()
	
	if flags.ownershipAssignment.lower() == "spectral":
		prefix = flags.inputDirectory
		outputFile = flags.outputDirectory
		gamma = flags.spectralGamma
		fuzzy_threshold = flags.fuzzy_threshold

		objectsDict = {}
		locationsDict = {}

		print("\n\n- - - - - Ingesting Data to Spectral Clustering - - - - - \n")
		for file in os.listdir(prefix):
			print("Adding",file,"to Spectral", f"File path: {prefix}/{file}")
			
			if file.startswith("objects"):
				with open(prefix+"/"+file, "r") as inObjs:
					objects = json.load(inObjs)
				objectsDict.update(objects)
			
			if file.startswith("locations"):
				with open(prefix+"/"+file, "r") as inLocs:
					locations = json.load(inLocs)
				locationsDict.update(locations)
		print("DONE Adding data to Spectral", f"len(locationsDict): {len(locationsDict)}", f"len(objectsDict): {len(objectsDict)}")
		
		ownerAssigner = OwnershipAssigner(locationsDict, objectsDict)
		df_locations, df_objects = ownerAssigner.convertToDataFrame(locationsDict, objectsDict)

		ownerAssigner._df_locations.to_csv(outputFile+"/locations.csv", index=False)
		ownerAssigner._df_objects.to_csv(outputFile+"/objects.csv", index=False)
	
		print("RUNNING SPECTRAL WITH N_CLUSTERS:", len(locationsDict.keys()), "GAMMA:", gamma)
		
		ownerAssigner.spectral_membership(n_clusters=len(locationsDict.keys()), gamma=gamma, fuzzy_threshold=fuzzy_threshold)

		ownerAssigner._df_objects.to_csv(outputFile+f"/SPECTRAL_PredictedLocations_N={len(locationsDict.keys())}_G={gamma}.csv", index=False)
		
		exit()
	
	if flags.ownershipAssignment.lower() == "seeded_iterative":
		prefix = flags.inputDirectory
		outputFile = flags.outputDirectory
		objectsDict = {}
		locationsDict = {}
		for file in os.listdir(prefix):
			if file.startswith("objects"):
				with open(prefix + "/" + file, "r") as inObjs:
					objects = json.load(inObjs)
					objectsDict.update(objects)
			if file.startswith("locations"):
				with open(prefix + "/" + file, "r") as inLocs:
					locations = json.load(inLocs)
					locationsDict.update(locations)
		ownerAssigner = OwnershipAssigner(locationsDict, objectsDict)
		df_locations, df_objects = ownerAssigner.convertToDataFrame(locationsDict, objectsDict)
		ownerAssigner._df_locations.to_csv(outputFile + "/locations.csv", index=False)
		ownerAssigner._df_objects.to_csv(outputFile + "/objects.csv", index=False)
		ownerAssigner.seeded_iterative_membership(
			n_nearest=flags.n_nearest,
        	noise_scale=2.0,
        	max_iters=50,
        	fuzzy_threshold=flags.fuzzy_threshold
		)
		ownerAssigner._df_objects.to_csv(
			outputFile + f"/SEEDED_ITERATIVE_PredictedLocations_FT={flags.fuzzy_threshold}.csv",
        	index=False
    	)
		exit()

	if flags.photoDownloader == True:
		start_wall_time = time.time()
		start_proc_time=time.process_time()
		b_box = flags.boundingbox
		print("BOUNDING BOX:",b_box)
		outputDirectory = flags.outputDirectory+str(b_box[0])+"_"+str(b_box[1])+"_"+str(b_box[2])+"_"+str(b_box[3])

		downer = PhotoDownloader()
		#UNCOMMENT 3 FOLLOWING LINES TO ENABLE API QUERYING
		photos, b_box_dict = downer.searchBoundingBox(b_box[0],b_box[1],b_box[2],b_box[3])
		downer.processQueryResults(photos,outputDirectory,page=flags.flickrPageNumber)

		json_file = outputDirectory+"/metadata"
		output = downer.detect_tags_from_jpgs_in_directory(outputDirectory, json_file)
		with open(json_file+"_objects.json", "w") as out:
			out.write(output)

		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO DOWNLOAD PHOTOS FROM FLICKR:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO DOWNLOAD PHOTOS FROM FLICKR:", end_wall_time-start_wall_time,"\n")

		exit()

	if flags.createConceptMaps==True: 

		start_wall_time = time.time()
		start_proc_time=time.process_time()
		
		predictedLocationsCSV = flags.inputFile
		outputDirectory = flags.outputDirectory
		locationsFile = flags.fileSource[0]
		CM = ConceptMapper()
		conceptMaps = CM.createConceptMap(predictedLocationsCSV)
		originalInputFile = predictedLocationsCSV
		predictedLocationsCSV = predictedLocationsCSV.split("/")[-1]
		predictedLocationsCSV = predictedLocationsCSV.replace(".csv","")

		outputFile = outputDirectory+"/ConceptMaps_"+predictedLocationsCSV+".pkl"
		
		try:
			os.makedirs(outputDirectory)
		except FileExistsError:
			pass
		with open(outputFile,"wb") as out:							#NB: Will need to use pickle to read back in.
			pickle.dump(conceptMaps ,out)

		## The Relative Location Dict:
		relativeLocationsDict = CM.createLocationCentricDict(inputFile=originalInputFile,locationsFile=locationsFile)

		outputFile = outputDirectory+"/RelativeLocations_"+predictedLocationsCSV+".JSON"

		with open (outputFile,"w") as outFile:
			json.dump(relativeLocationsDict, outFile, indent=4)
			print('DUMPED RELATIVE LOCATIONS DICT TO JSON')
		
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO CREATE COCNEPT MAPS:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO CREATE CONCEPT MAPS:", end_wall_time-start_wall_time,"\n")

		exit()

	

	if flags.gestaltSearch == True: 

		searchterms = flags.searchterms
		invertedIndexSourceCSV = flags.inputFile

		start_wall_time = time.time()
		start_proc_time=time.process_time()
		ii = InvertedIndex(invertedIndexSourceCSV)
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("\nPROCESSOR TIME TO GENERATE INVERTED INDEX:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN TO GENERATE INVERTED INDEX:", end_wall_time-start_wall_time,"\n")

		start_wall_time = time.time()
		start_proc_time=time.process_time()
		results = ii.search(searchterms)
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("SEARCH TERMS:", searchterms)
		if results:
			print("RETURNED STANDARD SEARCH RESULTS:", len(results[0]))	
		print("STANDARD II RESULTS ARE: ", results)
		print("\nPROCESSOR TIME FOR STANDARD QUERY:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN FOR STANDARD QUERY:", end_wall_time-start_wall_time,"\n")

		start_wall_time = time.time()
		start_proc_time=time.process_time()
		fuzzy_results = ii.fuzzy_search(searchterms)
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		
		print("SEARCH TERMS:", searchterms)
		if fuzzy_results[0]:
			print("RETURNED FUZZY SEARCH RESULTS:", len(fuzzy_results[0]))	
		print("FUZZY RESULTS ARE: ", fuzzy_results)
		print("\nPROCESSOR TIME FOR FUZZY QUERY:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN FOR FUZZY QUERY:", end_wall_time-start_wall_time,"\n")

		rankedResults = ii.ranked_search(searchterms)

		start_wall_time = time.time()
		start_proc_time=time.process_time()
		rankedResults = ii.ranked_search(searchterms)
		end_proc_time=time.process_time()
		end_wall_time = time.time()
		print("SEARCH TERMS:", searchterms)
		if rankedResults:
			print("RETURNED RANKED SEARCH RESULTS:", len(rankedResults[0]))	
		print("RANKED RESULTS ARE: ", rankedResults)
		print("\nPROCESSOR TIME FOR RANKED QUERY:",end_proc_time-start_proc_time)
		print("WALL TIME TAKEN FOR RANKED QUERY:", end_wall_time-start_wall_time,"\n")

		exit()


	if flags.dumpInvertedIndex == True:
		invertedIndexSourceCSV = flags.inputFile
		ii = InvertedIndex(invertedIndexSourceCSV)
		print("Inverted Index has", len(ii.ii.keys()), "object classes")
		for entry in ii.ii_counter:
			print(entry, ii.ii_counter[entry])
