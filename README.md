# SIQ-Anvil-Annotation-Tool


## How to launch the Anvil Backend: 


## Reproducing a demo: 

## Guide for Annotators:
### Step by Step Annotation Pipeline
1. Sign in to the annotation module. You will be directed to the main page.
2. Navigate between videos and questions using the navigation buttons on the bottom of the page.
3. View the assigned video.
4. Enter your question, correct answer, and incorrect answer in the text input box below the video.
5. Click the Save button to save your annotation.
6. The logits, attention map, and graphs will update.
7. If needed, edit your annotations and click Save when done.
8. You might want to repeat steps 6-7 in order to unbias your annotation.

### Understanding the Metrics:
* Logits: the decimal numbers represent the predicted probability of each sentence being the correct answer according to the model. Your goal is to minimize the difference between correct and incorrect answer (i.e. try to get the logits to be close to 0.5 for correct and incorrect answers)
* Attention Map: this heat map shows which words are most affecting the model. To edit and unbias your sentences, you might want to change the words with high attention value.
* Sentiment Graph: this graph shows the sentiment values of correct and incorrect answers. Your goal is to minimize the difference in sentiment at each question level and for the entire dataset.
* Emotion Bar Chart: in addition to the sentiment graph, this emotion bar chart can be used to determine where your correct and incorrect answer inputs fall on the negative to positve emotion scale. Your goal is to match the distribution between correct and incorrect answers.
* Polarity Graph: this graph shows the polarity values of correct and incorrect answers. Your goal is to minimize the difference in polarity at each question level and for the entire dataset.
* Intensity Graph: this graph shows the intensity values of correct and incorrect answers. Your goal is to minimize the difference in intensity at each question level and for the entire dataset.
* See detailed information about each metric in the **Available Metrics** section below.

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
* Logits: the logits are outputs of the RoBERTA model, which reveal the model's predictions of the percentage of each sentence being the correct answer
* Attention Map: the attention map is a heat map that corresponds to the attention values of each word in the correct and incorrect answer sentences. Higher attention value means that the word has more influence in the model, which is represented by a darker color on the heat map.
* Sentiment Graph: the NLTK library outputs the overal sentiment of an input sentence in range -1 to 1. The graph shows a t-test between correct answers (blue) and incorrect answers (red).
* Emotion Bar Chart: For each sentence, the SenticNet API outputs 1 or 2 emotions out of 24, which are divided by 6 emotion levels ranging from most negative to most positive. For each of the 6 emotion levels, the bar chart aggregates the number of sentences classified with that level.
* Polarity Graph: For ecah correct and incorrect answer pair, SenticNet API outputs one of the following labels: POSTIVE, NEGATIVE, or NEUTRAL. They correspond to 1, -1, and 0 on the graph respectively. Correct answers are colored blue and incorrect answers are colored red on the graph.
* Intensity Graph: For ecah correct and incorrect answer pair, SenticNet API outputs a number between -1 and 1 which represents a degree of emotions and sentiments. Correct answers are colored blue and incorrect answers are colored red on the graph.


### Backend: 
