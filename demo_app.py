import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from re import sub
import nltk
from nltk.corpus import stopwords
from PIL import Image
from io import BytesIO
from wordcloud import WordCloud, ImageColorGenerator
import sqlite3
import streamlit.components.v1 as components
import random

with st.echo(code_location='below'):
    df = pd.read_csv("goodreads_books.csv")
    df = df.dropna(subset=['genre_and_votes'])

    st.title("Книги с goodreads")

    st.subheader("Данных много, а Хероку не особо мощный, поэтому грузиться будет медленно. Спасибо за терпение")

    st.subheader("Нарисуем прикольную облачную картинку, которая будет показывать частотность слов из описания наших книг"
                 " (если немного внимательнее посмотреть, увидите, что окраска устроена таким способом, что вырисовывается силуэт книжной стопки)")

    nltk.download('stopwords')
    sw = stopwords.words('english')
    words = []

    #сделаем подготовку списка слов (заметьте, с использоавнием регулярных выражений :))
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

    #подгрузим маску в виде стопки книг, чтобы потом получить форму
    book_img='https://www.pinclipart.com/picdir/big/537-5379805_free-book-clipart-transparent-book-images-and-book.png'
    with urllib.request.urlopen(book_img) as url:
        f = BytesIO(url.read())
    img = Image.open(f)

    mask = np.array(img)
    img_color = ImageColorGenerator(mask)

    wc = WordCloud(background_color='white',
                   mask=mask,
                   max_font_size=150,
                   max_words=2000,
                   random_state=42)
    wcloud = wc.generate_from_frequencies(fr)
    fig=plt.figure(figsize=(10, 15))
    plt.axis('off')
    plt.imshow(wc.recolor(color_func=img_color), interpolation="bilinear")
    st.pyplot(fig)

    st.subheader("Давайте посмотрим на серии, в которых 10 книг")

    ser=set()
    for i in df['books_in_series']:
        if (type(i)==str) and (i.count(",") == 8):
            j=df[df['books_in_series']==i]['series'].to_string()
            j=j[j.find('(')+1:]
            j=j[:j.find('(')]
            j = j[:j.find(')')]
            try:
                j=j[:j.find('#')-1]
            except:pass
            if ord(j[3])<123:
                ser.add(j)

    ser=list(ser)
    ser1=list()
    for i in ser:
        k=0
        for j in df['series']:
            if i in str(j):
                k=k+1
        if k==10: ser1.append(i)

    selser = st.selectbox('Пожалуйста, выберите одну из серий из списка серий из 10 книг, которые остались после отсеивания из-за неполноты таблички',
                           ser1)

    serbook=list()
    for i in df['id']:
        if selser in df[df['id']==i]['series'].to_string():
            serbook.append(i)

    print(serbook)


    st.subheader("Подготовив данные (см. код внизу страницы) и с помощью SQL выгрузив их в R, построим график встречаемости 20 самых популярных жанров в ggplot2."
                 " Выгрузим html и посмотрим, что получилось")

    #выберем 20 самых популярных жанров
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
    c = conn.cursor()
    c.execute("""
    DROP TABLE IF EXISTS topgen;
    """)
    try:
        df1.to_sql(name='topgen', con=conn)
    except:
        pass
    conn.close()

    htmlf = open("rfile.nb.html", 'r', encoding='utf-8')
    source_code = htmlf.read()
    print(source_code)
    components.html(source_code, height=1200)