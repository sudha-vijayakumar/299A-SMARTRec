# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
#
# REFERENCES
# - https://medium.com/betacom/unsupervised-nlp-task-in-python-with-doc2vec-da1f7727857d
# - https://medium.com/betacom/building-a-rasa-chatbot-to-perform-listings-search-60cea9829e60
# - https://homes.cs.washington.edu/~msap/acl2020-commonsense/slides/02%20-%20knowledge%20in%20LMs.pdf
# - https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md

import os
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, utils
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.knowledge_base.actions import ActionQueryKnowledgeBase
from rasa_sdk.knowledge_base.storage import InMemoryKnowledgeBase

import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec 
from gensim.parsing.preprocessing import preprocess_string

import json
import torch
from bert_serving.client import BertClient
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import numpy as np
from sentence_transformers import SentenceTransformer

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

from py2neo import Graph

# sentence embedding selection.
# sentence_transformer_select=True
# pretrained_model='stsb-roberta-large' 
# score_threshold = 0.60  # This confidence scores can be adjusted based on your need!!
topK=5

# Load embedded features.
root = '/Users/sudhavijayakumar/Desktop/299/299A-SMARTRec'
embeddings = root+'/RASA/data/'
raw = root+'/Data/raw/'
# list_embedding_model = Doc2Vec.load(embeddings+'embeddings/list_embeddings')
# review_embedding_model = Doc2Vec.load(embeddings+'embeddings/review_embeddings')
# offline_models=root+'RASA/offline_models/'

# Load dataset to get listings and reviews.
listings = pd.read_csv(raw+'listings.csv.gz', sep=',', usecols = ['id','listing_url','picture_url','name','description','neighbourhood','property_type','bedrooms','bathrooms','amenities','price','review_scores_rating']) 
reviews = pd.read_csv(raw+'reviews.csv.gz', sep=',', usecols = ['listing_id','comments']) 

# use model to find the listings
# spark = SparkSession.builder.getOrCreate()
# sc = spark.sparkContext
# alsmodel = ALSModel.load(offline_models+"als_model")

USE_NEO4J = bool(os.getenv("USE_NEO4J", True))

if USE_NEO4J:
    from neo4j_knowledge_base import Neo4jKnowledgeBase

# use neo4j for real-time recommendations.
g = Graph("bolt://localhost:7687/neo4j", password = "test")
user_id="10952" #to be replaced with dynamic value.

def realTimeRecommendation(topK):		

	query = """
							// Get count of all distinct products that user 4789 has purchased and find other users who have purchased them
							MATCH (u1:User {name:$uid})-[x:RATED]->(m:Listing)<-[y:RATED]-(u2:User)
							WHERE u1 <> u2
							WITH u1, u2, COUNT(DISTINCT m) as intersection_count
							
							// Get count of all the distinct products that are unique to each user
							MATCH (u:User)-[:RATED]->(m:Listing)
							WHERE u in [u1, u2]
							WITH u1, u2, intersection_count, COUNT(DISTINCT m) as union_count
						
							// Compute Jaccard index
							WITH u1, u2, intersection_count, union_count, (intersection_count*1.0/union_count) as jaccard_index
							
							// Get top k neighbours based on Jaccard index
							ORDER BY jaccard_index DESC, u2.id
							WITH u1, COLLECT([u2.name, jaccard_index, intersection_count, union_count])[0..$k] as neighbours
							RETURN u1.name as user, neighbours
							"""
	neighbours = {}
	for row in g.run(query, uid=user_id, k=topK):
		neighbours[row[0]] = row[1]

	neighbours_ids = [x[0] for x in neighbours[row[0]]]

	return neighbours_ids

neighbours_ids = realTimeRecommendation(5)

# # Recommendations based on feature embeddings.
# class ActionlistingsDetails_Embedding(Action):

# 	def name(self) -> Text:
# 		return "action_listings_details_embedding"

# 	def run(self, dispatcher: CollectingDispatcher,
# 			tracker: Tracker,
# 			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

# 		userMessage = tracker.latest_message['text']
# 		# use model to find the listings
# 		new_doc = preprocess_string(userMessage)
# 		test_doc_vector = list_embedding_model.infer_vector(new_doc)
# 		sims = list_embedding_model.docvecs.most_similar(positive = [test_doc_vector])		
		
# 		# Get first 5 matches
# 		for s in sims[:1]:
# 			picture = listings['picture_url'].iloc[s[0]]
# 			listingss = listings['listing_url'].iloc[s[0]]
# 			name = listings['name'].iloc[s[0]]
# 			description = listings['description'].iloc[s[0]]
# 			neighbourhood = listings['neighbourhood'].iloc[s[0]]
# 			bedroom = listings['bedrooms'].iloc[s[0]]
# 			bathroom = listings['bathrooms'].iloc[s[0]]
# 			amenities = listings['amenities'].iloc[s[0]]
# 			price = listings['price'].iloc[s[0]]
# 			review_score_rating = listings['review_scores_rating'].iloc[s[0]]

# 		botResponse = "Please find the top listing details:\nlink: "+str(listingss)+"\nTitle: "+str(name)+"\nDescription: "+str(description)+"\nNeighbourhood: "+str(neighbourhood)+"\nBedroom: "+str(bedroom)+"\nBathroom: "+str(bathroom)+"\nAmenities: "+str(amenities)+"\nPrice: "+str(price)+" per night\nRating: "+str(review_score_rating)+" on a scale of 5"
		
# 		dispatcher.utter_message(text=botResponse)
# 		dispatcher.utter_message(image=picture)

# 		return []

# class ActionlistingsSearch_Embedding(Action):

# 	def name(self) -> Text:
# 		return "action_listings_search_embedding"

# 	def run(self, dispatcher: CollectingDispatcher,
# 			tracker: Tracker,
# 			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

# 		userMessage = tracker.latest_message['text']
# 		# use model to find the listings
# 		new_doc = preprocess_string(userMessage)
# 		test_doc_vector = list_embedding_model.infer_vector(new_doc)
# 		sims = list_embedding_model.docvecs.most_similar(positive = [test_doc_vector])		
		
# 		# Get first 5 matches
# 		listingss = [listings['listing_url'].iloc[s[0]] for s in sims[:5]]

# 		botResponse = f"Here are the listing details: {listingss}.".replace('[','').replace(']','')
		
# 		dispatcher.utter_message(text=botResponse)

# 		return []

# class ActionlistingsPics_Embedding(Action):

# 	def name(self) -> Text:
# 		return "action_listings_pics_embedding"

# 	def run(self, dispatcher: CollectingDispatcher,
# 			tracker: Tracker,
# 			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

# 		userMessage = tracker.latest_message['text']

# 		new_doc = preprocess_string(userMessage)
# 		test_doc_vector = list_embedding_model.infer_vector(new_doc)
# 		sims = list_embedding_model.docvecs.most_similar(positive = [test_doc_vector])		
		
# 		listingss = [listings['picture_url'].iloc[s[0]] for s in sims[:1]]

# 		str = ''
# 		for lst in listingss:
# 			str = lst

# 		botResponse = f"Your requirement seems to match with the following listing:"
		
# 		dispatcher.utter_message(text=botResponse)
# 		dispatcher.utter_message(image=str)

# 		return []

class ActionlistingsBook(Action):

	def name(self) -> Text:
		return "action_book"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		botResponse = f"Booking confirmed under your account!"
		
		dispatcher.utter_message(text=botResponse)

		return []

class ActionlistingsCancel(Action):

	def name(self) -> Text:
		return "action_cancel"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		botResponse = f"Booking cancelled!"
		
		dispatcher.utter_message(text=botResponse)

		return []

# class ActionGetHelpLink_Embedding(Action):

#     def __init__(self):
#         super(ActionGetHelpLink_Embedding, self).__init__()
#         self.faq_data = json.load(open(raw+"faq.json", "rt", encoding="utf-8"))
#         self.sentence_embedding_choose(sentence_transformer_select, pretrained_model)
#         self.standard_questions_encoder = np.load(embeddings+"embeddings/questions_embedding.npy")
#         self.standard_questions_encoder_len = np.load(embeddings+"embeddings/questions_embedding_len.npy")
#         print(self.standard_questions_encoder.shape)

#     def sentence_embedding_choose(self, sentence_transformer_select=True, pretrained_model='stsb-roberta-large'):
#         self.sentence_transformer_select = sentence_transformer_select
#         if sentence_transformer_select:
#             self.bc = SentenceTransformer(pretrained_model)
#         else:
#             self.bc = BertClient(check_version=False)

#     def get_most_similar_standard_question_id(self, query_question):
#         if self.sentence_transformer_select:
#             query_vector = torch.tensor(self.bc.encode([query_question])[0]).numpy()
#         else:
#             query_vector = self.bc.encode([query_question])[0]
#         print("Question received at action engine")
#         score = np.sum((self.standard_questions_encoder * query_vector), axis=1) / (
#                 self.standard_questions_encoder_len * (np.sum(query_vector * query_vector) ** 0.5))
#         top_id = np.argsort(score)[::-1][0]
#         return top_id, score[top_id]

#     def name(self) -> Text:
#         return "action_get_HelpLink_Embedding"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         query = tracker.latest_message['text']
#         print(query)
#         most_similar_id, score = self.get_most_similar_standard_question_id(query)
#         print("The question is matched with id:{} with score: {}".format(most_similar_id,score))
#         if float(score) > score_threshold: # This confidence scores can be adjusted based on your need!!
#             dispatcher.utter_message('Finding help, please wait!')
#             dispatcher.utter_message('Please refer the below help link for more informaion:')
#             response = self.faq_data[most_similar_id]['a']
#             dispatcher.utter_message(response)
#             dispatcher.utter_message("Did it help?")
#         else:
#             response = "Sorry, I don't see any matched help :("
#             dispatcher.utter_message(response)
#             dispatcher.utter_message("Sorry, I can't answer your question. You can dial customer support +1-844-234-2500")
#         return []

# class ActionlistingsReviews_Embedding(Action):

# 	def name(self) -> Text:
# 		return "action_listings_reviews_Embedding"

# 	def run(self, dispatcher: CollectingDispatcher,
# 			tracker: Tracker,
# 			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

# 		userMessage = tracker.latest_message['text']
# 		# use model to find the listings
# 		new_doc = preprocess_string(userMessage)
# 		test_doc_vector = review_embedding_model.infer_vector(new_doc)
# 		sims = review_embedding_model.docvecs.most_similar(positive = [test_doc_vector])		
		
# 		# Get first 5 matches
# 		listingss=[]
# 		for s in sims:
# 			listingss.append('https://www.airbnb.com/rooms/'+str(s[0]))

# 		botResponse = f"Here are the top recommendation for you: {listingss}.".replace('[','').replace(']','')
				
# 		dispatcher.utter_message(text=botResponse)

# 		return []

# ## Recommendations based on collaborative filtering.
# class ActionlistingsDetails_CBF(Action):
# 	def name(self) -> Text:
# 		return "action_listings_details_cbf"

# 	def run(self, dispatcher: CollectingDispatcher,
# 			tracker: Tracker,
# 			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

# 		userMessage = tracker.latest_message['text']
# 		userMessage='2818' ##(to-be-replaced) with dynamic value
# 		# use model to find the listings
# 		model = Word2Vec.load(offline_models+'ContentBasedFilter')		
# 		sims = model.wv.similar_by_vector(userMessage, topK+1)[1:]
# 		# Get first 5 matches
# 		listingss=[]
# 		for s in sims:
# 			listingss.append('https://www.airbnb.com/rooms/'+s[0])

# 		botResponse = f"Here are the top recommendation for you: {listingss}.".replace('[','').replace(']','')
		
# 		dispatcher.utter_message(text=botResponse)

# 		return []

## Recommendations based on collaborative filtering.
# class ActionlistingsDetails_ColabF(Action):
# 	def name(self) -> Text:
# 		return "action_listings_details_colabf"

# 	def run(self, dispatcher: CollectingDispatcher,
# 			tracker: Tracker,
# 			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

# 		userMessage = tracker.latest_message['text']
	
# 		# convert this into a dataframe so that it can be passed into the recommendForUserSubset
		
# 		user_id = [[164729]] ##(to-be-replaced) with dynamic value
# 		functiondf = sc.parallelize(user_id).toDF(['reviewer_id'])

# 		recommendations = alsmodel.recommendForUserSubset(functiondf , topK)
# 		recommendations.collect()

# 		sims = [recommendations.collect()[0]['recommendations'][x]['listing_id'] for x in range(0,topK)]
# 		# Get first 5 matches
# 		listingss=[]
# 		for s in sims:
# 			listingss.append('https://www.airbnb.com/rooms/'+str(s))

# 		botResponse = f"Here are the top recommendation for you: {listingss}.".replace('[','').replace(']','')
		
# 		dispatcher.utter_message(text=botResponse)

# 		return []

## Recommendations based on real-time collaborative filtering.
class ActionlistingsDetails_Neo4jColabF(Action):
	def name(self) -> Text:
		return "action_listings_details_neo4j_colabf"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
			
		query = """
							// Get top n recommendations for user from the selected neighbours
							MATCH (u1:User),
								(neighbour:User)-[:RATED]->(p:Listing)        // get all listings rated by neighbour
							WHERE u1.name = $uid
							AND neighbour.id in $neighbours
							AND not (u1)-[:RATED]->(p)                        // which u1 has not already bought
							
							WITH u1, p, COUNT(DISTINCT neighbour) as cnt                                // count times rated by neighbours
							ORDER BY u1.name, cnt DESC                                               // and sort by count desc
							RETURN u1.name as user, COLLECT([p.name,cnt])[0..$k] as recos  
							"""

		recos = {}
		for row in g.run(query, uid=user_id, neighbours=neighbours_ids, k=topK):
			recos[row[0]] = row[1]

		sims = [x[0] for x in recos[row[0]]]   

		# Get first 5 matches
		listingss=[]
		for s in sims:
			listingss.append('https://www.airbnb.com/rooms/'+str(s))

		botResponse = f"Here are the top recommendation for you: {listingss}.".replace('[','').replace(']','')
		
		dispatcher.utter_message(text=botResponse)

		return []

## Recommendations based on real-time content-based filtering.
class ActionlistingsDetails_Neo4jCBF(Action):
	def name(self) -> Text:
		return "action_listings_details_neo4j_cbf"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
			
		query = """
				MATCH (u:User {name:$cid})-[:RATED]->(s:Listing)-[:HAS_AMENITY]->(c:Amenity)<-[:HAS_AMENITY]-(z:Listing)
				WHERE NOT EXISTS ((u)-[:RATED]->(z))
				WITH s, z, COUNT(c) AS intersection
				MATCH (s)-[:HAS_AMENITY]->(sc:Amenity)
				WITH s, z, intersection, COLLECT(sc.name) AS s1
				MATCH (z)-[:HAS_AMENITY]->(zc:Amenity)
				WITH s, z, s1, intersection, COLLECT(zc.name) AS s2
				WITH s, z, intersection, s1+[x IN s2 WHERE NOT x IN s1] AS union, s1, s2
				RETURN s.name as UserListing, z.name as Recommendate, s1 as UserListingAmenities, s2 as RecommendateListingAmenities, ((1.0*intersection)/SIZE(union)) AS jaccard ORDER BY jaccard DESC LIMIT $k;
				"""
		listingss=[]
		recoAmenity=[]
		for row in g.run(query, cid = "8726758", k = 5).data():
			listingss.append('https://www.airbnb.com/rooms/'+str(row['Recommendate']))
			recoAmenity.append(str(row['UserListingAmenities']))

		botResponse = f"Here are the top recommendation for you: {listingss}.".replace('[','').replace(']','')
		dispatcher.utter_message(text=botResponse)

		return []


class Neo4jKnowledgeBaseAction(ActionQueryKnowledgeBase):
    def name(self) -> Text:
        return "action_response_query"

    def __init__(self):
        if USE_NEO4J:
            print("using Neo4jKnowledgeBase")
            knowledge_base = Neo4jKnowledgeBase(
                "bolt://localhost:7687", "neo4j", "test"
            )
        else:
            print("using InMemoryKnowledgeBase")
            #query locally.

        super().__init__(knowledge_base)

    async def utter_objects(
        self,
        dispatcher,
        object_type,
        objects,
    ) -> None:
        """
        Utters a response to the user that lists all found objects.
        Args:
            dispatcher: the dispatcher
            object_type: the object type
            objects: the list of objects
        """
        if objects:
            dispatcher.utter_message(text=f"Found the following {object_type}s:")

            repr_function = await utils.call_potential_coroutine(
                self.knowledge_base.get_representation_function_of_object(object_type)
            )

            for i, obj in enumerate(objects, 1):
                dispatcher.utter_message(text=f"{i}: {repr_function(obj)}")
        else:
            dispatcher.utter_message(text=f"I didn't find any {object_type}s.")

    def utter_attribute_value(
        self,
        dispatcher,
        object_name,
        attribute_name,
        attribute_value,
    ) -> None:
        """
        Utters a response that informs the user about the attribute value of the
        attribute of interest.
        Args:
            dispatcher: the dispatcher
            object_name: the name of the object
            attribute_name: the name of the attribute
            attribute_value: the value of the attribute
        """
        if attribute_value:
            dispatcher.utter_message(
                text=f"The {attribute_name} of {object_name} is {attribute_value}."
            )
        else:
            dispatcher.utter_message(
                text=f"I didn't find the {attribute_name} of {object_name}."
            )
