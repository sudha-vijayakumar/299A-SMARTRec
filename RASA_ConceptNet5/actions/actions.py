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
		#replace with dynamic value.
		prediction = tracker.latest_message['entities'][0]['value']
		
		if prediction:
			#replace with dynamic value.
			word=str(prediction)
			word=word.lower()
			
			if word in mem_cache_conceptNet5:
				collection = mem_cache_conceptNet5[word]

				# print('Recommendation based on the following review tags:')
				# print(collection)
				## query string

				query_string="MATCH (r:Review_Text)-[]-(l:Listing) WHERE "
				for item in collection:
					query_string+="r.name=~'(?i).*"+item.lower()+".*' or "
				query_string+=" r.name=~'(?i).*"+word.lower()+".*'"
				query_string+=" RETURN l.url As url,l.picture_url As pic LIMIT "+str(topK)+";"


				query = ""+query_string+""
				for row in g.run(query, query_string=query_string,k=topK):
					dispatcher.utter_message(text=row[0].replace('\'',''))
					dispatcher.utter_message(image=row[1])
			else:
				query_string="MATCH (r:Review_Text)-[]-(l:Listing) WHERE "
				query_string+=" r.name=~'(?i).*"+word.lower()+".*'"
				query_string+=" RETURN l.url As url,l.picture_url As pic LIMIT "+str(topK)+";"


				query = ""+query_string+""
				for row in g.run(query, query_string=query_string,k=topK):
					dispatcher.utter_message(text=row[0].replace('\'',''))
					dispatcher.utter_message(image=row[1])

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
		
		prediction = tracker.latest_message['entities'][0]['value']
		
		if prediction:
			#replace with dynamic value.
			word=str(prediction)
			word=word.lower()
			
			# print('Recommendation based on the following review tags:')
			# print(collection)

			if word in mem_cache_conceptNet5:
				collection = mem_cache_conceptNet5[word]
				print(collection)
				## query string
				query_string="MATCH (r:Listing_Text)-[]-(l:Listing) WHERE "
				for item in collection:
					query_string+="r.name=~'(?i).*"+item.lower()+".*' or "
				query_string+=" r.name=~'(?i).*"+word.lower()+".*'"
				query_string+=" RETURN l.url As url,l.picture_url As pic LIMIT "+str(topK)+";"


				query = ""+query_string+""
				for row in g.run(query, query_string=query_string,k=topK):
					dispatcher.utter_message(text=row[0].replace('\'',''))
					dispatcher.utter_message(image=row[1])
			else:
				query_string="MATCH (r:Listing_Text)-[]-(l:Listing) WHERE "
				query_string+=" r.name=~'(?i).*"+word.lower()+".*'"
				query_string+=" RETURN l.url As url,l.picture_url As pic LIMIT "+str(topK)+";"
				query = ""+query_string+""
				for row in g.run(query, query_string=query_string,k=topK):
					dispatcher.utter_message(text=row[0].replace('\'',''))
					dispatcher.utter_message(image=row[1])

		else:
			dispatcher.utter_message(text="No matched listings")

		return []