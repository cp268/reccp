import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

import config, rec_sys
from ingredient_parser import ingredient_parser

import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


def make_clickable(name, link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = name
    return f'<a target="_blank" href="{link}">{text}</a>'


def main():
    image = Image.open("input/image.jpg").resize((680, 150))
    st.image(image)
    st.markdown("# *Need a recipe for your ingredients?*")

    st.markdown(
        "Use this ML powered app for a recipe recommendation! <a href='https://github.com/cp268/reccp' > <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/600px-Octicons-mark-github.svg.png' width='20' height='20' > </a> ",
        unsafe_allow_html=True,
    )

    st.markdown(
        "Our app is built from a collection of over 120,000 recipes! Type in your list of ingredients below and get the top recipe recommendations with instructions!"
    )
    st.text("")

    #session_state = st.session_state.get(
        #model_computed=False,
        #execute_recsys=False,
    #)

    ingredients = st.text_input("Enter ingredients you would like to cook with")
    st.session_state.execute_recsys = st.button("Give me recommendations!")

    if st.session_state.execute_recsys:

        col1, col2, col3 = st.beta_columns([1, 6, 1])
        with col2:
            gif_runner = st.image("input/cooking_gif.gif")
        recipe = rec_sys.RecSys(ingredients)
        gif_runner.empty()
        st.session_state.recipe_df_clean = recipe.copy()
        # link is the column with hyperlinks
        recipe["url"] = recipe.apply(
            lambda row: make_clickable(row["recipe"], row["url"]), axis=1
        )
        recipe_display = recipe[["recipe", "url", "ingredients"]]
        st.session_state.recipe_display = recipe_display.to_html(escape=False)
        st.session_state.recipes = recipe.recipe.values.tolist()
        st.session_state.model_computed = True
        st.session_state.execute_recsys = False

    if st.session_state.model_computed=True:
        # st.write("Either pick a particular recipe or see the top 5 recommendations.")
        recipe_all_box = st.selectbox(
            "Either see the top 5 recommendations or see the top selection",
            ["Show me them all!", "Select a single recipe"],
        )
        if recipe_all_box == "Show me them all!":
            st.write(st.session_state.recipe_display, unsafe_allow_html=True)
        else:
            selection = st.selectbox(
                "Select a delicious recipe", options=st.session_state.recipes
            )
            selection_details = st.session_state.recipe_df_clean.loc[
                st.session_state.recipe_df_clean.recipe == selection
            ]
            st.write(f"Recipe: {selection_details.recipe.values[0]}")
            st.write(f"Ingredients: {selection_details.ingredients.values[0]}")
            st.write(f"URL: {selection_details.url.values[0]}")
            st.write(f"Score: {selection_details.score.values[0]}")



if __name__ == "__main__":
    main()
