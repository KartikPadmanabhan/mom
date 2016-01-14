from bokeh.plot_object import PlotObject
from bokeh.server.utils.plugins import object_page
from bokeh.server.app import bokeh_app
from bokeh.plotting import curdoc, cursession
from bokeh.crossfilter.models import CrossFilter
from bokeh.sampledata.autompg import autompg
from IPython.display import Image
import pandas as pd
import numpy as np

df=pd.read_csv('data/TMP_AWR_LOAD_PROFILE_AGG.csv', sep='|',parse_dates=True)
df=df.pivot_table(index='start_time', columns='stat_name', values='stat_per_sec')
cols=df.columns
for col in cols:
    meancol=np.mean(df[col])
    df[col]=df[col].fillna(meancol)
@bokeh_app.route("/bokeh/crossfilter/")
@object_page("crossfilter")
def make_crossfilter():
    app = CrossFilter.create(df=df)
    return app
