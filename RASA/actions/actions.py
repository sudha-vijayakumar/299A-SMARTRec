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


from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.parsing.preprocessing import preprocess_string

import json
import torch
from bert_serving.client import BertClient
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import numpy as np
from sentence_transformers import SentenceTransformer

# sentence embedding selection
sentence_transformer_select=True
pretrained_model='stsb-roberta-large' # Refer: https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md
score_threshold = 0.60  # This confidence scores can be adjusted based on your need!!
# Load ML model
root = '/Users/sudhavijayakumar/Documents/299/299A-SMARTRec/RASA/data'

model = Doc2Vec.load(root+'embeddings/list_embeddings')
review_model = Doc2Vec.load(root+'embeddings/review_embeddings')

# Load dataset to get listings titles
df = pd.read_csv(root+'listings.csv.gz', sep=',', usecols = ['listing_url','picture_url','name','description','neighbourhood','property_type','bedrooms','bathrooms','amenities','price','review_scores_rating']) 
reviews = pd.read_csv(root+'reviews.csv.gz', sep=',', usecols = ['listing_id','comments']) 

class ActionlistingsDetails(Action):

	def name(self) -> Text:
		return "action_listings_details"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
		# use model to find the listings
		new_doc = preprocess_string(userMessage)
		test_doc_vector = model.infer_vector(new_doc)
		sims = model.dv.most_similar(positive = [test_doc_vector])		
		
		# Get first 5 matches
		for s in sims[:1]:
			picture = df['picture_url'].iloc[s[0]]
			listingss = df['listing_url'].iloc[s[0]]
			name = df['name'].iloc[s[0]]
			description = df['description'].iloc[s[0]]
			neighbourhood = df['neighbourhood'].iloc[s[0]]
			bedroom = df['bedrooms'].iloc[s[0]]
			bathroom = df['bathrooms'].iloc[s[0]]
			amenities = df['amenities'].iloc[s[0]]
			price = df['price'].iloc[s[0]]
			review_score_rating = df['review_scores_rating'].iloc[s[0]]

		botResponse = "Please find the top listing details:\nlink: "+str(listingss)+"\nTitle: "+str(name)+"\nDescription: "+str(description)+"\nNeighbourhood: "+str(neighbourhood)+"\nBedroom: "+str(bedroom)+"\nBathroom: "+str(bathroom)+"\nAmenities: "+str(amenities)+"\nPrice: "+str(price)+" per night\nRating: "+str(review_score_rating)+" on a scale of 5"
		
		dispatcher.utter_message(text=botResponse)
		dispatcher.utter_message(image=picture)

		return []

class ActionlistingsSearch(Action):

	def name(self) -> Text:
		return "action_listings_search"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
		# use model to find the listings
		new_doc = preprocess_string(userMessage)
		test_doc_vector = model.infer_vector(new_doc)
		sims = model.dv.most_similar(positive = [test_doc_vector])		
		
		# Get first 5 matches
		listingss = [df['listing_url'].iloc[s[0]] for s in sims[:5]]

		botResponse = f"Here are the listing details: {listingss}.".replace('[','').replace(']','')
		
		dispatcher.utter_message(text=botResponse)

		return []

class ActionlistingsPics(Action):

	def name(self) -> Text:
		return "action_listings_pics"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']

		new_doc = preprocess_string(userMessage)
		test_doc_vector = model.infer_vector(new_doc)
		sims = model.dv.most_similar(positive = [test_doc_vector])		
		
		listingss = [df['picture_url'].iloc[s[0]] for s in sims[:1]]

		str = ''
		for lst in listingss:
			str = lst

		botResponse = f"Your requirement seems to match with the following listing:"
		
		dispatcher.utter_message(text=botResponse)
		dispatcher.utter_message(image=str)

		return []

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

class ActionGetFAQAnswer(Action):

    def __init__(self):
        super(ActionGetFAQAnswer, self).__init__()
        self.faq_data = json.load(open(root+"faq.json", "rt", encoding="utf-8"))
        self.sentence_embedding_choose(sentence_transformer_select, pretrained_model)
        self.standard_questions_encoder = np.load(root+"questions_embedding.npy")
        self.standard_questions_encoder_len = np.load(root+"questions_embedding_len.npy")
        print(self.standard_questions_encoder.shape)

    def sentence_embedding_choose(self, sentence_transformer_select=True, pretrained_model='stsb-roberta-large'):
        self.sentence_transformer_select = sentence_transformer_select
        if sentence_transformer_select:
            self.bc = SentenceTransformer(pretrained_model)
        else:
            self.bc = BertClient(check_version=False)

    def get_most_similar_standard_question_id(self, query_question):
        if self.sentence_transformer_select:
            query_vector = torch.tensor(self.bc.encode([query_question])[0]).numpy()
        else:
            query_vector = self.bc.encode([query_question])[0]
        print("Question received at action engine")
        score = np.sum((self.standard_questions_encoder * query_vector), axis=1) / (
                self.standard_questions_encoder_len * (np.sum(query_vector * query_vector) ** 0.5))
        top_id = np.argsort(score)[::-1][0]
        return top_id, score[top_id]

    def name(self) -> Text:
        return "action_get_FAQ"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.latest_message['text']
        print(query)
        most_similar_id, score = self.get_most_similar_standard_question_id(query)
        print("The question is matched with id:{} with score: {}".format(most_similar_id,score))
        if float(score) > score_threshold: # This confidence scores can be adjusted based on your need!!
            dispatcher.utter_message('Finding help, please wait!')
            dispatcher.utter_message('Please refer the below help link for more informaion:')
            response = self.faq_data[most_similar_id]['a']
            dispatcher.utter_message(response)
            dispatcher.utter_message("Did it help?")
        else:
            response = "Sorry, I don't see any matched help :("
            dispatcher.utter_message(response)
            dispatcher.utter_message("Sorry, I can't answer your question. You can dial customer support +1-844-234-2500")
        return []

class ActionlistingsReviews(Action):

	def name(self) -> Text:
		return "action_listings_reviews"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
		# use model to find the listings
		new_doc = preprocess_string(userMessage)
		test_doc_vector = review_model.infer_vector(new_doc)
		sims = review_model.dv.most_similar(positive = [test_doc_vector])		
		
		# Get first 5 matches
		for s in sims[:1]:
			listing_id = df['listing_id'].iloc[s[0]]
			matched = df[df['listing_id']==listing_id]

			picture = matched['picture_url']
			listingss = matched['listing_url']
			name = matched['name']
			description = matched['description']
			neighbourhood = matched['neighbourhood']
			bedroom = matched['bedrooms']
			bathroom = matched['bathrooms']
			amenities = matched['amenities']
			price = matched['price']
			review_score_rating = matched['review_scores_rating']
			
		botResponse = "Please find the top listing details that matches with your expectations:\nlink: "+str(listingss)+"\nTitle: "+str(name)+"\nDescription: "+str(description)+"\nNeighbourhood: "+str(neighbourhood)+"\nBedroom: "+str(bedroom)+"\nBathroom: "+str(bathroom)+"\nAmenities: "+str(amenities)+"\nPrice: "+str(price)+" per night\nRating: "+str(review_score_rating)+" on a scale of 5"
		
		dispatcher.utter_message(text=botResponse)
		dispatcher.utter_message(image=picture)

		return []
