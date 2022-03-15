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
import re
import gensim
from gensim.models import Word2Vec 
from gensim.parsing.preprocessing import preprocess_documents
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time 

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

topK=5

# Load embedded features.
root = '.'
embeddings = root+'/RASA_realtime_recommendation/data/'
raw = root+'/Data/raw/'

USE_NEO4J = bool(os.getenv("USE_NEO4J", True))

if USE_NEO4J:
    from neo4j_knowledge_base import Neo4jKnowledgeBase

# use neo4j for real-time recommendations.
g = Graph("bolt://localhost:7687/neo4j", password = "test")
user_id="10952" #can be replaced with dynamic userid value.
print("Connected to Neo4j")

from collections import defaultdict

# neighbours_ids = []
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

neighbours_ids = realTimeRecommendation(20)

## Recommendations based on real-time collaborative filtering.
class ActionlistingsDetails_Neo4jColabF(Action):
	def name(self) -> Text:
		return "action_listings_details_neo4j_colabf"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
			
		print(userMessage)

		# neighbours_ids = realTimeRecommendation(20)

		botResponse = f"Gettings similar real-time recommendations..."
		dispatcher.utter_message(text=botResponse)

	

		query = """
							// Get top n recommendations for user from the selected neighbours
							MATCH (u1:User),
								(neighbour:User)-[:RATED]->(p:Listing)        // get all listings rated by neighbour
							WHERE u1.name = $uid
							AND neighbour.id in $neighbours
							AND not (u1)-[:RATED]->(p)                        // which u1 has not already bought
							
							WITH u1, p, COUNT(DISTINCT neighbour) as cnt                                // count times rated by neighbours
							ORDER BY u1.name, cnt DESC                                               // and sort by count desc
							RETURN u1.name as user, COLLECT([p.name,p.picture_url,p.accomodates,p.bathrooms,p.bedrooms,p.beds,p.host_identity_verified,p.review_scores_rating,p.price,cnt])[0..$k] as recos  
							"""

		recos = {}

		botResponse = f"Here are the top recommendation for you:"
		dispatcher.utter_message(text=botResponse)
		for row in g.run(query, uid=user_id, neighbours=neighbours_ids, k=topK):
			recos[row[0]] = row[1]

		print(recos)

		data = {"payload": 'cardsCarousel'}
		image_list = []
		count=0
		for row in recos[row[0]]:
			dic={}
			dic["image"] = str(row[1])
			dic['title'] = str(row[0])
			dic['url'] = 'https://www.airbnb.com/rooms/'+str(row[0])
					
			image_list.append(dic)

			dispatcher.utter_message(text='https://www.airbnb.com/rooms/'+str(row[0]))
			dispatcher.utter_message(text="Accomodates:"+str(row[2]))
			dispatcher.utter_message(text="Bedrooms:"+str(row[4]))
			dispatcher.utter_message(text="Bathrooms:"+str(row[3]))
			dispatcher.utter_message(text="Beds:"+str(row[5]))
			dispatcher.utter_message(text="Host_Verified:"+str(row[6]))
			dispatcher.utter_message(text="Score:"+str(row[7]))
			dispatcher.utter_message(text="Price:"+str(row[8]))
			dispatcher.utter_message(image=str(row[1]))
			dispatcher.utter_message(text="\n***")
			count+=1

		if count==0:
			dispatcher.utter_message(text="no great matches! Can you rephrase?")
		else:
			data["data"]=image_list
			dispatcher.utter_message(json_message=data)

		return []

## Recommendations based on real-time collaborative filtering.
class ActionlistingsDetails_Neo4jColabFExclude(Action):
	def name(self) -> Text:
		return "action_listings_details_neo4j_colabf_exclude"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
			
		print(userMessage)

		# neighbours_ids = realTimeRecommendation(20)

		botResponse = f"Gettings similar real-time recommendations..."
		dispatcher.utter_message(text=botResponse)

		query = """
							// Get top n recommendations for user from the selected neighbours
							MATCH (u1:User),
								(neighbour:User)-[:RATED]->(p:Listing)        // get all listings rated by neighbour
							WHERE u1.name = $uid
							AND neighbour.id in $neighbours
							AND not (u1)-[:RATED]->(p)                        // which u1 has not already bought
							
							WITH u1, p, COUNT(DISTINCT neighbour) as cnt                                // count times rated by neighbours
							ORDER BY u1.name, cnt DESC                                               // and sort by count desc
							RETURN u1.name as user, COLLECT([p.name,p.picture_url,p.accomodates,p.bathrooms,p.bedrooms,p.beds,p.host_identity_verified,p.review_scores_rating,p.price,cnt])[$k..$n] as recos  
							"""

		recos = {}

		botResponse = f"Here are the top recommendation for you:"
		dispatcher.utter_message(text=botResponse)

		topK = 6	
		topN = topK + 5

		for row in g.run(query, uid=user_id, neighbours=neighbours_ids, k=topK,n=topN):
			recos[row[0]] = row[1]

		print(recos)

		data = {"payload": 'cardsCarousel'}
		image_list = []
		count=0
		for row in recos[row[0]]:
			dic={}
			dic["image"] = str(row[1])
			dic['title'] = str(row[0])
			dic['url'] = 'https://www.airbnb.com/rooms/'+str(row[0])
					
			image_list.append(dic)

			dispatcher.utter_message(text='https://www.airbnb.com/rooms/'+str(row[0]))
			dispatcher.utter_message(text="Accomodates:"+str(row[2]))
			dispatcher.utter_message(text="Bedrooms:"+str(row[4]))
			dispatcher.utter_message(text="Bathrooms:"+str(row[3]))
			dispatcher.utter_message(text="Beds:"+str(row[5]))
			dispatcher.utter_message(text="Host_Verified:"+str(row[6]))
			dispatcher.utter_message(text="Score:"+str(row[7]))
			dispatcher.utter_message(text="Price:"+str(row[8]))
			dispatcher.utter_message(image=str(row[1]))
			dispatcher.utter_message(text="\n***")
			count+=1

		if count==0:
			dispatcher.utter_message(text="no great matches! Can you rephrase?")
		else:
			data["data"]=image_list
			dispatcher.utter_message(json_message=data)

		return []

## Recommendations based on real-time content-based filtering.
class ActionlistingsDetails_Neo4jCBF(Action):
	def name(self) -> Text:
		return "action_listings_details_neo4j_cbf"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
		
		print(userMessage)

		query = """
				MATCH (u:User {name:$cid})-[:RATED]->(s:Listing)-[:HAS_AMENITY]->(c:Amenity)<-[:HAS_AMENITY]-(z:Listing)
				WHERE NOT EXISTS ((u)-[:RATED]->(z))
				WITH s, z, COUNT(c) AS intersection
				MATCH (s)-[:HAS_AMENITY]->(sc:Amenity)
				WITH s, z, intersection, COLLECT(sc.name) AS s1
				MATCH (z)-[:HAS_AMENITY]->(zc:Amenity)
				WITH s, z, s1, intersection, COLLECT(zc.name) AS s2
				WITH s, z, intersection, s1+[x IN s2 WHERE NOT x IN s1] AS union, s1, s2
				RETURN s.name as UserListing, z.name as Recommendate, z.picture_url as url, z.accomodates as accomodates,z.bathrooms as bathrooms,z.bedrooms as bedrooms,z.beds as beds,z.host_identity_verified as verified,z.review_scores_rating as review_scores,z.price as price,s1 as UserListingAmenities, s2 as RecommendateListingAmenities, ((1.0*intersection)/SIZE(union)) AS jaccard ORDER BY jaccard DESC LIMIT $k;
				"""
		listingss=[]
		recoAmenity=[]
		
		botResponse = f"Here are the top recommendation for you"
		dispatcher.utter_message(text=botResponse)

		data = {"payload": 'cardsCarousel'}
		image_list = []
		count=0
		for row in g.run(query, cid = "8726758", k = 5).data():
			print(row)
			dic={}
			dic["image"] = str(row['url'])
			dic['title'] = str(row['Recommendate'])
			dic['url'] = 'https://www.airbnb.com/rooms/'+str(row['Recommendate'])
			image_list.append(dic)
			dispatcher.utter_message(text='https://www.airbnb.com/rooms/'+str(row['Recommendate']))
			dispatcher.utter_message(text="Amenities:\n")
			dispatcher.utter_message(text=str(row['UserListingAmenities']))
			dispatcher.utter_message(text="Accomodates:"+str(row['accomodates']))
			dispatcher.utter_message(text="Bedrooms:"+str(row['bedrooms']))
			dispatcher.utter_message(text="Bathrooms:"+str(row['bathrooms']))
			dispatcher.utter_message(text="Beds:"+str(row['beds']))
			dispatcher.utter_message(text="Host_Verified:"+str(row['verified']))
			dispatcher.utter_message(text="Score:"+str(row['review_scores']))
			dispatcher.utter_message(text="Price:"+str(row['price']))
			dispatcher.utter_message(image=str(row['url']))
			dispatcher.utter_message(text="\n***")
			count+=1

		if count==0:
			dispatcher.utter_message(text="no great matches! Can you rephrase?")
		else:
			data["data"]=image_list
			print(image_list)
			print(data)
			dispatcher.utter_message(json_message=data)

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
                if object_type=="Listing":
                    obj = "https://www.airbnb.com/rooms/"+repr_function(obj)
                    dispatcher.utter_message(text=f"{obj}")
                else:
                    dispatcher.utter_message(text=f"{obj}")
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
