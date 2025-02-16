# WhoIsTalking
Discover what lays behind political speeches. "Who is talking?" analyzes a text and shows you which party it most likely would belong to. Simply insert the text, analyze it and get the results in seconds.
This [poster](https://github.com/julietteyek/WhoIsTalking/blob/main/Poster_whoistalking.pdf) summarises all the important information and results.

## <ins>Inhaltsverzeichnis</ins>

## <ins>Motivation</ins>

Wouldn't it be interesting to know what political affiliation can be inferred from a text? And that from a more neutral observer than humans?

"Who is talking?" aims to create a tool that analyzes language in terms of its political orientation and reliably estimates its political direction.

Beyond personal amusement—like analyzing a family chat message and sending the result back without comment—this project has broader benefits:

- Uncovering the political bias of articles and news

- Analyzing propagandistic language and determining its characteristics

- Analyzing the evolution of parliamentary party sentiments towards topics

This project represents the first stage of this application. Many envisioned developments (see Outlook) are yet to be implemented.



## <ins>Method</ins>

...



## <ins>Installation & Usage</ins>

###A. Prerequisites

Ensure you have Python 3.8+ and install the required dependencies:

pip install -r requirements.txt

###B. Running the Application

You can test the model on a sample text via the streamlit app by running:

streamlit run application.py



## <ins>Outlook</ins>

Multilingual Expansion: Including multiple languages and dialects for international scalability.

Topic Sentiment Analysis: Providing deeper insights into political discourse by analyzing how different parties frame topics like healthcare or immigration.

Real-time Evaluation: Implementing an API for live text evaluation in political debates or social media discussions.

Media Bias Assessment: Extending the tool to assess political bias in public media content and evaluate audience engagement, such as analyzing sentiment in YouTube comments.



## <ins>Acknowledgements</ins>

This project utilizes BERT-base-uncased and is inspired by NLP research in political speech analysis. Special thanks to Ana Calotescu for providing most of the datasets used for this application.


For questions, contributions, or discussions, feel free to open an issue or reach out!
