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
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from robobrowser import RoboBrowser
import requests
from bs4 import BeautifulSoup
import imageio as io
import base64
from numpy import polynomial as P
from sklearn.linear_model import LinearRegression

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

    #сделаем подготовку списка слов (заметьте, с использованием регулярных выражений :))
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

    st.subheader("Давайте посмотрим на серии, в которых 10 книг (эта часть, к сожалению, грузится особенно долго)")

    #создадим сет из серий, которые претендуют на то, что в них 10 книг
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

    #оставим только те, в которых действительно 10 книг
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

    #список книг серии
    serbook=list()
    for i in df['title']:
        if selser in df[df['title']==i]['series'].to_string():
            serbook.append(i)

    #создадим список ссылок на обложки, используя робобраузер
    images=list()
    for i in serbook:
        browser = RoboBrowser(history=True)
        browser.open('https://www.goodreads.com/list/show/1.Best_Books_Ever')
        form = browser.get_form(action='/search')
        form['q'].value = i
        browser.submit_form(form)
        page = browser.select('.bookTitle')
        browser.follow_link(page[0])
        b = str(browser)
        lin=b[b.find('=')+1: b.find('>')]
        r=requests.get(lin).text
        s=BeautifulSoup(r)
        for j in s.find_all("img"):
            try:
                if j['id']=='coverImage':
                    images.append(j['src'])
                    break
            except:
                pass

    st.subheader("Ловите гифку с обложками книг из серии")

    with io.get_writer('posters.gif', mode='I', duration=0.5) as writer:
        for i in range (10):
            image = io.imread(images[i])
            writer.append_data(image)
    writer.close()
    file_ = open("posters.gif", "rb")
    contents = file_.read()
    url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{url}" alt="gif">',
        unsafe_allow_html=True
    )

    st.subheader("Посмотрим, как меняется рейтинг от книги к книгк в серии 'Akiko Books' "
                 "(она просто очень удобно лежит в этой табличке, книги идут в том же порядке, что и в серии)")

    #список рейтингов
    rat=list()
    for i in df['series']:
        try:
            if 'Akiko Books' in i:
                rat.append(float(df[df['series']==i]['average_rating']))
        except: pass

    st.subheader("Используя питоновскую математику, а именно интерполяцию многочленов (линал:)) и немного машинки, посмотрим,"
                 " имеет ли смысл пытаться точно оценивать функцию изменения рейтинга в этом случае или сойдёт обычная"
                 " линейная регрессия")

    #построим интерполяционные многочлены 9, 15 и 20 степеней
    x=np.array([1,2,3,4,5,6,7,8,9,10])
    y=np.array(rat)
    f1 = P.Polynomial.fit(x, y, 9)
    f2 = P.Polynomial.fit(x, y, 15)
    f3 = P.Polynomial.fit(x, y, 20)

    #построим линейную регрессию
    x=x.reshape((-1, 1))
    regr=LinearRegression().fit(x, y)
    def f4(x):
        return x*regr.coef_+regr.intercept_

    #теперь построим график и посмотрим, насколько многочлчен далёк от прямой
    xx = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(xx, f1(xx), color='black', lw=2,  label='Интерполяция 9 степени')
    ax.plot(xx, f2(xx), color='blue', lw=2, label='Интерполяция 15 степени')
    ax.plot(xx, f3(xx), color='green', lw=2, label='Интерполяция 20 степени')
    ax.plot(xx, f4(xx), 'r--', lw=2, label='Линейная регрессия')
    ax.scatter(x, y, label='Данные по рейтингу')
    ax.legend(loc=0)
    ax.set_xticks(x)
    ax.set_xlabel("номер книги", fontsize=18)
    ax.set_ylabel("рейтинг", fontsize=18)
    st.pyplot(fig)

    st.subheader("Видно, что прямая не так сильно отдалена от наших точек, а интерполяционные многочлены периодически"
                 " уходят в какие-то странные стороны. Ещё раз вспоминаем о том, что иногда лучше недоучить, "
                 "чем переучить модель")


    st.subheader("Подготовив данные (см. код внизу страницы) и с помощью SQL выгрузив их в R,"
                 " построим график встречаемости 20 самых популярных жанров в ggplot2."
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