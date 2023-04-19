# SIQ-Anvil-Annotation-Tool


## How to launch the Anvil Backend: 


## Reproducing a demo: 


## General documentation: 

### Frontend: 
The frontend of the annotation tool was created using anvil. The primary function of the tool is to allow annotators to watch videos and generate questions and answers based on these videos. We also display various bias metrics, which help the annotator remove semantic biases in their annotations.

#### Features:
* Header: the header at the top contains the video id and question number of the current page
* Video: this is the embedded YouTube video that we want to annotate
* Question/Answer Textboxes: there are three textboxes (question, correct answer, incorrect answer) that takes in the annotator's inputs
* Graph Section: this section contains multiple graphs that show distributions of various metrics that can be used by the annotator to visualize the bias in their annotation
* Navigation buttons: annotator can navigate between videos and questions using the navigation buttons. The save button allows the annotator to save current input to the database, which also updates the graphs.

#### Available Metrics:
* Logits:
* Attention Map:
* Sentiment Graph:
* Emotion Bar Chart:
* Subjectivity Graph:
* Intensity Graph:


### Backend: 
