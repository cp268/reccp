# Recipe Recommendation System
## Project Overview
* This is a tool that recommends recipes based on ingredients that the user inputs.
* Scraped over 120,000 recipes from Food Network, All Recipes, and Epicureus.
* Parsed recipe ingredients and created word embeddings using Word2Vec and TF-IDF.
* Created a recipe recommendation system using cosine similarity to measure Euclidean distance between the word embeddings of recipe ingredients.
* Used Streamlit Cloud to deploy my app: https://share.streamlit.io/cp268/reccp/main/streamlit.py (ignore the error message and type the ingredients)

## Code 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, gensim, matplotlib, seaborn, beautifulsoup, flask, streamlit, json, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  


## Web Scraping
For each recipe, the following was obtained:

* Recipe name 
* Ingredients
* URL
* Rating

## Repo organization
```
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── recsys.py               <- app to run project / user interface
├── scripts                 <- directory for pipeline scripts or utility scripts
    ├── ingredient_parser.py<- script to parse ingredients
    ├── tfidf_encoder.py    <- script to perform embedding aggregation
    ├── word2vec_rec.py     <- script to perform word embeddings
├── models                  <- directory for pkl files for trained models
├── data                    <- directory for project data
    ├── raw                 <- directory for raw data or script to download
    ├── processed           <- directory to store processed data
    ├── outputs             <- directory to store any output data
├── notebooks               <- directory to store any exploration notebooks used
    ├──recipe_compile.ipynb <- compiles recipe name, url, and ingredients from the three sources of recipe 
├── streamlit.py            <- launch streamlit app on streamlit cloud
├── .gitignore              <- git ignore file
```

## Data Cleaning
After scraping the data, we needed to parse the ingredients to remove redundant information that would not help distinguish recipes. The **ingredient parser** does the following:
* Lemmatize words to ensure we remove all versions of words, e.g. both pounds and pound
* Removed stopwords 
* Removed cooking measures, e.g. pounds and lbs
* Removed common household items, such as oil and butter 
* Some standard NLP preprocessing: getting rid of punctuation, removing accents, making everything lowercase, getting rid of Unicode, etc.
 
## Model Building
I fed my collection of individual ingredients into a **continuous bag of words Word2Vec** neural network to produce word embeddings. Word2Vec was used because I wanted the text representations to capture distributional similarities between words. In the context of recipe ingredients, Word2vec allowed me to capture similarities between recipe ingredients that are commonly used together.

In order to build the recipe recommendation system, I needed to represent each ingredient list as a single embedding as this would then allow me to calculate corresponding similarities. **TF-IDF** was used to aggregate embeddings (as opposed to simple averaging) as it gives us better distinguishing power between recipes by favoring unique ingredients.

The recommendation system was built using a **content-based filtering** approach which enables us to recommend recipes to people based on the ingredients the user provides. To measure the similarity between user-given ingredients and recipes **cosine similarity** was used. The recommendation model computes the cosine similarity between the inputted ingredient list and all recipes in the corpus. It then outputs the top-N most similar recipes, along with their ingredients and URLs, for the user to choose from.
