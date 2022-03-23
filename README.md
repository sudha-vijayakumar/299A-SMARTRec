# 299A-SMARTREC_A-SMART-Conversational-Recommendation-System-using-Semantic-Knowledge-Graph

## FRONT-END DEMO
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/1_h9_2ar29I/0.jpg)](https://www.youtube.com/watch?v=1_h9_2ar29I)

## BACK-END DEMO
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/33FOCLEwvbM/0.jpg)](https://youtu.be/33FOCLEwvbM)

# Solution Architecture
![alt text](https://github.com/sudha-vijayakumar/299A-SMARTRec/blob/main/Documents/report_diagrams/Architecture/ARCHITECTURE-HIGH.png)


## RUNTIME
Prerequisites: 
  Before, getting started install the following,
  - Python 3 => https://docs.python-guide.org/starting/install3/osx/
  - Jupyter notebook/ MS Visual Studio Code => https://jupyter.org/install
  - RASA => https://rasa.com/docs/rasa/installation/
  - Neo4J Desktop 
  
## STEP-1: DATA GATHERING & PREPROCESSING

> Execute the *.ipynb in the numerical order. 
> Skip unused scripts.


## STEP-2: Setting up Airbnb domain knowledge graph and ConceptNet5 in Neo4J,
https://github.com/sudha-vijayakumar/299A-SMARTRec/tree/main/Neo4J_QUERIES


## STEP-3: HOW TO THE SOLUTION 

### START RASA
Run the below commands in separate terminals,

Terminal-1:
- $ rasa train
- $ rasa run -m models --enable-api --cors "*" --debug

Terminal-2:
- $ rasa run actions

### START NEO4J FROM THE NEO4J DESKTOP 

### RUN CHATBOT WIDGET
   - Unzip ChatBot_Widget folder.
   - Hit open ChatBot_Widget/index.html to start interacting with SMARTREC
