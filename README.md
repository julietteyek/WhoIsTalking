![alt text](https://github.com/julietteyek/WhoIsTalking/blob/main/logo_who_is_talkig.png "Logo_Who_is_talking?")

"Who is Talking?" is a project that seeks to leverage machine learning and natural language processing (NLP) techniques to classify political speech based on its ideological leaning. By analyzing linguistic patterns and contextual cues, the project aims to provide insights into how different political groups communicate and frame their messages.This tool analyzes a text and shows you which party it most likely would belong to. Simply insert the text, analyze it and get the results. Trained only with German texts.\
[This poster](https://github.com/julietteyek/WhoIsTalking/blob/main/Who_is_talking.pdf) summarises all the important information and results.

## Inhaltsverzeichnis

1. [Motivation](##Motivation)
2. [Installation & Usage](##Installation-&-Usage)
3. [Outlook](##Outlook)
4. [Acknowledgements](##Acknowledgements)

## Motivation

Wouldn't it be interesting to know what political affiliation can be inferred from a text? And that from a more neutral observer than humans?

"Who is talking?" aims to create a tool that analyzes language in terms of its political orientation and reliably estimates its political direction.

Beyond personal amusement — like analyzing a family chat message and sending the result back without comment — this project aims to provide broader benefits like:

- Uncovering the political bias of articles and news

- Analyzing propagandistic language and determining its characteristics

- Analyzing the evolution of parliamentary party sentiments towards topics

This project represents the first stage of this application. Many envisioned developments (see Outlook) are yet to be implemented.



## Installation & Usage

###  A. Prerequisites

First: Clone this repository and ideally create either a new environment or an envionment where torch is working conflict-free.
Ensure you have Python 3.8+ and install the required dependencies from the WhoIsTalking folder:

```bash
pip install -r src/requirements.txt
```

### B. Running the Application

You can test the model on a text of your choice (in German) via the streamlit app by running:

``` bash 
streamlit run application.py
```
in the src-folder.

The current Model "bundestag-speeches-lite" is uploaded on Huggingface and is accessed in my code with
``` 
model = BertForSequenceClassification.from_pretrained("julietteyek/bundestag-speeches-lite")
```
🔒 No inputs are being saved as for this version. 🔒
## Outlook

(First, I will need to speed up the analysis process a lot, I know :D)

Multilingual Expansion: Including multiple languages and dialects for international scalability.

Topic Sentiment Analysis: Providing deeper insights into political discourse by analyzing how different parties frame topics like healthcare or immigration.

Real-time Evaluation: Implementing an API for live text evaluation in political debates or social media discussions.

Media Bias Assessment: Extending the tool to assess political bias in public media content and evaluate audience engagement, such as analyzing sentiment in YouTube comments.



## Acknowledgements

This project utilizes BERT-base-uncased and is inspired by NLP research in political speech analysis.\
Special thanks to Ana Calotescu for providing most of the datasets used for this application.


*For questions, contributions, or discussions, feel free to open an issue or reach out!*
