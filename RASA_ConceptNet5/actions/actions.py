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

import json

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from py2neo import Graph
from collections import defaultdict

# sentence embedding selection.
# sentence_transformer_select=True
# pretrained_model='stsb-roberta-large' 
# score_threshold = 0.60  # This confidence scores can be adjusted based on your need!!
topK=5

# Load mem_cache features.

mem_cache_conceptNet5={}
with open('mem_cache_conceptNet5.json') as json_file:
    mem_cache_conceptNet5 = json.load(json_file)

# use neo4j for real-time recommendations.
g = Graph("bolt://localhost:7687/neo4j", password = "test")

print("Connected to Neo4j")

#used to retrieve data from conceptNet5 in real-time.
def getConceptTags(word,topK):		
	collection = []
	query = """
				CALL ga.nlp.ml.word2vec.nn($wid, $k, 'en-ConceptNet5') YIELD word, distance RETURN word AS list;
			"""
	for row in g.run(query, wid=word, k=topK):
			processed=re.sub('[^a-zA-Z0-9]+', ' ', row[0])
			collection.append(processed)
	return collection

## Recommendations based on real-time conceptnet + review text fusion.
class ActionReview_ConceptNet5(Action):
	def name(self) -> Text:
		return "action_reviews_ConceptNet5"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
		
		print(userMessage)

		data = {"payload": 'cardsCarousel'}
		image_list = []
			
		#replace with dynamic value.
		prediction = tracker.latest_message['entities'][0]['value']
		
		if prediction:
			#replace with dynamic value.
			word=str(prediction)
			word=word.lower()

			if word in mem_cache_conceptNet5:
				collection = mem_cache_conceptNet5[word]


				query_string="MATCH (r:Review_Text)-[]-(l:Listing) WHERE "
				tags=""
				for item in collection:
					query_string+="r.name=~'(?i).*"+item.lower()+".*' or "
					tags+=item.lower()+","
				query_string+=" r.name=~'(?i).*"+word.lower()+".*'"
				query_string+=" RETURN l.name as name, l.url As url,l.picture_url As picture_url,l.accomodates as accomodates,l.bathrooms as bathrooms,l.bedrooms as bedrooms,l.beds as beds,l.host_identity_verified as verified,l.review_scores_rating as review_scores,l.price as price LIMIT "+str(topK)+";"
				
				query = ""+query_string+""
				count=0
				for row in g.run(query, query_string=query_string,k=topK):
					print(row)
					dic={}
					dic["image"] = row['picture_url']
					dic['title'] = row['name']
					dic['url'] = row['url']
					
					image_list.append(dic)

					dispatcher.utter_message(text=str(row['url']))
					dispatcher.utter_message(text="Accomodates:"+str(row['accomodates']))
					dispatcher.utter_message(text="Bedrooms:"+str(row['bedrooms']))
					dispatcher.utter_message(text="Bathrooms:"+str(row['bathrooms']))
					dispatcher.utter_message(text="Beds:"+str(row['beds']))
					dispatcher.utter_message(text="Host_Verified:"+str(row['verified']))
					dispatcher.utter_message(text="Price:"+str(row['price']))
					dispatcher.utter_message(image=str(row['picture_url']))
					dispatcher.utter_message(text="\n***")
					count+=1
				
				if count==0:
					dispatcher.utter_message(text="no great matches! Can you rephrase?")
				else:
					dispatcher.utter_message(text='Recommendation based on the following review tags:')
					dispatcher.utter_message(text=tags.rstrip(','))
					data["data"]=image_list

					dispatcher.utter_message(json_message=data)
			else:
				query_string="MATCH (r:Review_Text)-[]-(l:Listing) WHERE "
				query_string+=" r.name=~'(?i).*"+word.lower()+".*'"
				query_string+=" RETURN l.name as name, l.url As url,l.picture_url As picture_url,l.accomodates as accomodates,l.bathrooms as bathrooms,l.bedrooms as bedrooms,l.beds as beds,l.host_identity_verified as verified,l.review_scores_rating as review_scores,l.price as price LIMIT "+str(topK)+";"
				query = ""+query_string+""
				count=0
				for row in g.run(query, query_string=query_string,k=topK):
					print(row)
					dic={}
					dic["image"] = row['picture_url']
					dic["title"] = row['name']
					dic["url"] = row['url']
					
					image_list.append(dic)


					dispatcher.utter_message(text=str(row['url']))
					dispatcher.utter_message(text="Accomodates:"+str(row['accomodates']))
					dispatcher.utter_message(text="Bedrooms:"+str(row['bedrooms']))
					dispatcher.utter_message(text="Bathrooms:"+str(row['bathrooms']))
					dispatcher.utter_message(text="Beds:"+str(row['beds']))
					dispatcher.utter_message(text="Host_Verified:"+str(row['verified']))
					dispatcher.utter_message(text="Price:"+str(row['price']))
					dispatcher.utter_message(image=str(row['picture_url']))
					dispatcher.utter_message(text="\n***")
					count+=1
				if count==0:
					dispatcher.utter_message(text="no great matches! Can you rephrase?")
				else:
					dispatcher.utter_message(text='Recommendation based on the following review tags:')
					dispatcher.utter_message(text=word)
					data["data"]=image_list

					dispatcher.utter_message(json_message=data)

		else:
			dispatcher.utter_message(text="No matched listings")

		return []

class ActionListing_ConceptNet5(Action):
	def name(self) -> Text:
		return "action_listing_ConceptNet5"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
		
		print(userMessage)

		prediction = tracker.latest_message['entities'][0]['value']
		
		data = {"payload": 'cardsCarousel'}
		image_list = []

		if prediction:
			#replace with dynamic value.
			word=str(prediction)
			word=word.lower()
			
			if word in mem_cache_conceptNet5:
				collection = mem_cache_conceptNet5[word]
				
				## query string
				tags=""
				query_string="MATCH (r:Listing_Text)-[]-(l:Listing) WHERE "
				for item in collection:
					query_string+="r.name=~'(?i).*"+item.lower()+".*' or "
					tags+=item.lower()+","
				query_string+=" r.name=~'(?i).*"+word.lower()+".*'"
				query_string+=" RETURN l.name as name, l.url As url,l.picture_url As picture_url,l.accomodates as accomodates,l.bathrooms as bathrooms,l.bedrooms as bedrooms,l.beds as beds,l.host_identity_verified as verified,l.review_scores_rating as review_scores,l.price as price LIMIT "+str(topK)+";"


				query = ""+query_string+""
				count=0
				for row in g.run(query, query_string=query_string,k=topK):
					print(row)
					dic={}
					dic["image"] = row['picture_url']
					dic["title"] = row['name']
					dic["url"] = row['url']
					image_list.append(dic)

					dispatcher.utter_message(text=str(row['url']))
					dispatcher.utter_message(text="Accomodates:"+str(row['accomodates']))
					dispatcher.utter_message(text="Bedrooms:"+str(row['bedrooms']))
					dispatcher.utter_message(text="Bathrooms:"+str(row['bathrooms']))
					dispatcher.utter_message(text="Beds:"+str(row['beds']))
					dispatcher.utter_message(text="Host_Verified:"+str(row['verified']))
					dispatcher.utter_message(text="Price:"+str(row['price']))
					dispatcher.utter_message(image=str(row['picture_url']))
					dispatcher.utter_message(text="\n***")
					count+=1
				if count==0:
					dispatcher.utter_message(text="no great matches! Can you rephrase?")
				else:
					dispatcher.utter_message(text='Recommendation based on the following listing tags:')
					dispatcher.utter_message(text=tags.rstrip(','))
					data["data"]=image_list
			else:
				print('Recommendation based on the following listing tags:')
				print(word)
				query_string="MATCH (r:Listing_Text)-[]-(l:Listing) WHERE "
				query_string+=" r.name=~'(?i).*"+word.lower()+".*'"
				query_string+=" RETURN l.name as name, l.url As url,l.picture_url As picture_url,l.accomodates as accomodates,l.bathrooms as bathrooms,l.bedrooms as bedrooms,l.beds as beds,l.host_identity_verified as verified,l.review_scores_rating as review_scores,l.price as price LIMIT "+str(topK)+";"
				query = ""+query_string+""
				count=0
				for row in g.run(query, query_string=query_string,k=topK):
					print(row)
					dic={}
					dic["image"] = row['picture_url']
					dic["title"] = row['name']
					dic["url"] = row['url']
					image_list.append(dic)

					dispatcher.utter_message(text=str(row['url']))
					dispatcher.utter_message(text="Accomodates:"+str(row['accomodates']))
					dispatcher.utter_message(text="Bedrooms:"+str(row['bedrooms']))
					dispatcher.utter_message(text="Bathrooms:"+str(row['bathrooms']))
					dispatcher.utter_message(text="Beds:"+str(row['beds']))
					dispatcher.utter_message(text="Host_Verified:"+str(row['verified']))
					dispatcher.utter_message(text="Price:"+str(row['price']))
					dispatcher.utter_message(image=str(row['picture_url']))
					dispatcher.utter_message(text="\n***")

					count+=1
				if count==0:
					dispatcher.utter_message(text="no great matches! Can you rephrase?")
				else:
					dispatcher.utter_message(text='Recommendation based on the following listing tags:')
					dispatcher.utter_message(text=word)
					data["data"]=image_list
				
		else:
			dispatcher.utter_message(text="No matched listings")

		return []


class ActionImageCarosaul(Action):
	def name(self) -> Text:
		return "action_image_carosaul"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']

		data= [ { "title": "Sick Leave", "description": "Sick leave is time off from work that workers can use to stay home to address their health and safety needs without losing pay." }, { "title": "Earned Leave", "description": "Earned Leaves are the leaves which are earned in the previous year and enjoyed in the preceding years. " }, { "title": "Casual Leave", "description": "Casual Leave are granted for certain unforeseen situation or were you are require to go for one or two days leaves to attend to personal matters and not for vacation." }, { "title": "Flexi Leave", "description": "Flexi leave is an optional leave which one can apply directly in system at lease a week before." } ]
		message={ "payload": "collapsible", "data": data }

		dispatcher.utter_message(text="You can apply for below leaves",json_message=message)
				

		return []