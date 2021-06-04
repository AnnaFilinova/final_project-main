import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from re import sub, match
import nltk
from nltk.corpus import stopwords
from PIL import Image
from io import BytesIO
from wordcloud import WordCloud, ImageColorGenerator, get_single_color_func
import sqlite3
import streamlit.components.v1 as components

with st.echo(code_location='below'):
    df = pd.read_csv("goodreads_books.csv")
    df = df.dropna(subset=['genre_and_votes'])

    #посмотрим, какие жанры у нас вообще есть и выберем 20 самых популярных
    genres=set()
    d=dict()
    for i in df['genre_and_votes']:
        for j in i.split(','):
            l=''
            for k in j.split()[:-1]:
                l=l+' '+k
            genres.add(l.strip())
            try:
                d[l.strip()] = d[l.strip()] + 1
            except:
                d[l.strip()] = 1
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    d=dict(d[:20])
    topgen=list(d.keys())

    #теперь перекинем наш словарь с жанрам и их встречаемостью в SQL, чтобы открыть в R и построить график там
    df1=pd.DataFrame(d.items(), columns=['genre', 'instances'])
    conn = sqlite3.connect('topgen.sqlite')
    df1.to_sql(name='topgen', con=conn)
    conn.close()

    st.title("Книги с goodreads")

    st.subheader("Подготовив данные и с помощью SQL выгрузив их в R, строим график в ggplot2. Выгрузим html и посмотрим, что получилось")

    htmlf = open("rfile.nb.html", 'r', encoding='utf-8')
    source_code = htmlf.read()
    print(source_code)
    components.html(source_code, height=2000)

    nltk.download('stopwords')
    sw = stopwords.words('english')
    words = []

    for text in df["description"]:
        if type(text) == type(float("nan")):
            continue
        text = text.lower()
        text = sub(r'\[.*?\]', '', text)
        text = sub(r'([.!,?])', r' \1 ', text)
        text = sub(r'[^a-zA-Z.,!?]+', r' ', text)
        text = [i for i in text.split() if i not in sw]
        for word in text:
            words.append(word)

    fr = nltk.FreqDist([i for i in words if len(i) > 2])

    #book_img = 'https://www.pinclipart.com/picdir/middle/365-3651885_book-black-and-white-png-peoplesoft-learn-peoplesoft.png'
    book_img='https://www.pinclipart.com/picdir/big/537-5379805_free-book-clipart-transparent-book-images-and-book.png'
    with urllib.request.urlopen(book_img) as url:
        f = BytesIO(url.read())
    img = Image.open(f)

    mask = np.array(img)
    img_color = ImageColorGenerator(mask)
    #img_color= get_single_color_func('deepskyblue')

    wc = WordCloud(background_color='white',
                   mask=mask,
                   max_font_size=1500,
                   max_words=2000,
                   random_state=42)
    wcloud = wc.generate_from_frequencies(fr)
    fig=plt.figure(figsize=(10, 15))
    plt.axis('off')
    plt.imshow(wc.recolor(color_func=img_color), interpolation="bilinear")
    st.pyplot(fig)


