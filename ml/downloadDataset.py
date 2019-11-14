import requests
import sqlite3
import pandas as pd
import os

database = '../../Huereka/bam.sqlite'
style = 'media_oilpaint'
location = './' + style + '/'

if not os.path.exists(location):
    os.makedirs(location)

db = sqlite3.connect(database)
query = """SELECT m.src, m.mid FROM modules AS m, automatic_labels AS a
           WHERE m.mid == a.mid AND a.{0} == 'positive'""".format(style)
df = pd.read_sql(query, db)
for url, mid in zip(df['src'], df['mid']):
    img = requests.get(url).content
    filename = location + str(mid) + '.jpg'
    with open(filename, 'wb+') as f:
        f.write(img)
