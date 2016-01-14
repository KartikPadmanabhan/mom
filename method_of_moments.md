
# USING METHOD OF MOMENTS TO PREDICT DATABASE PERFORMANCE METRICS

## Introduction

This article exploits statistical technique method of moments (MOM) and generalized method of moments (GMM)  to estimate database statistics from the past data. Both GMM and MOM's are parametric  unsupervised form of learning. I am restricting this to Oracle Database in this example, but it can be extended to any database system.

We initially select a distribution to model our random variable. Our random variables in database statistical parameters could be any statistic that we collect say in AWR report. To start our analysis, we collected AWR reports over several months. We then scraped these data and stored our metrics in csv formats. I have presented some csv files corresponding to these statistics in my github page.


## Parameters Of Interest

Lets say we want to use the method of moments to predict the following AWR statistics for the database Load Profile in AWR statistics. This is given in this section.


    %matplotlib inline
    from IPython.display import Image
    import pandas as pd
    import random
    import matplotlib.pyplot as plt
    from scipy.stats.kde import gaussian_kde
    from plotly import tools
    import scipy.stats as scs
    from scipy.stats import *
    from scipy import stats
    import plotly.plotly as py
    from plotly.graph_objs import *
    import numpy as np
    pd.set_option('display.max_columns', 10)


    Image(filename='images/load_profile.png')




![png](method_of_moments_files/method_of_moments_6_0.png)



Reading our historic load profile data in the dataframe.


    df=pd.read_csv('data/TMP_AWR_LOAD_PROFILE_AGG.csv', sep='|',parse_dates=True)
    df.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>site_id</th>
      <th>config_id</th>
      <th>run_id</th>
      <th>stat_id</th>
      <th>stat_name</th>
      <th>stat_per_sec</th>
      <th>stat_per_txn</th>
      <th>stat_per_exec</th>
      <th>stat_per_call</th>
      <th>start_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>265418</td>
      <td>Global Cache blocks received</td>
      <td>4467.21</td>
      <td>16.36</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>03-OCT-14 08.00.00.000000 AM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>265427</td>
      <td>Transactions</td>
      <td>274.49</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>03-OCT-14 08.00.00.000000 AM</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>265410</td>
      <td>Logical read (blocks)</td>
      <td>291880.19</td>
      <td>1064.87</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>03-OCT-14 08.00.00.000000 AM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>265419</td>
      <td>Global Cache blocks served</td>
      <td>4467.02</td>
      <td>16.34</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>03-OCT-14 08.00.00.000000 AM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>265421</td>
      <td>Parses (SQL)</td>
      <td>1131.04</td>
      <td>4.14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>03-OCT-14 08.00.00.000000 AM</td>
    </tr>
  </tbody>
</table>
</div>



As we can we have the data for several statistics viz. Global Cache blocks received, Transactions, Logical read (blocks), Global Cache blocks served, etc from the AWR load profile start from date 03-OCT-14  until 02-NOV-15.
I am interested in forecasting only stat_per_sec information on this AWR Load profile, so using pivot function in pandas we get the following reconstructed dataframe.      


    df=df.pivot_table(index='start_time', columns='stat_name', values='stat_per_sec')
    df.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>stat_name</th>
      <th>Background CPU(s)</th>
      <th>Block changes</th>
      <th>DB CPU(s)</th>
      <th>DB Time(s)</th>
      <th>Executes (SQL)</th>
      <th>...</th>
      <th>Session Logical Read IM</th>
      <th>Transactions</th>
      <th>User calls</th>
      <th>Write IO (MB)</th>
      <th>Write IO requests</th>
    </tr>
    <tr>
      <th>start_time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>01-APR-15 08.00.00.000000 AM</th>
      <td>2.34</td>
      <td>7171.66</td>
      <td>8.34</td>
      <td>21.33</td>
      <td>7295.85</td>
      <td>...</td>
      <td>NaN</td>
      <td>342.13</td>
      <td>3172.36</td>
      <td>7.38</td>
      <td>791.23</td>
    </tr>
    <tr>
      <th>01-APR-15 10.00.00.000000 PM</th>
      <td>1.88</td>
      <td>5829.02</td>
      <td>5.75</td>
      <td>14.98</td>
      <td>4943.24</td>
      <td>...</td>
      <td>NaN</td>
      <td>243.01</td>
      <td>2288.04</td>
      <td>5.59</td>
      <td>603.88</td>
    </tr>
    <tr>
      <th>01-JAN-15 08.00.00.000000 AM</th>
      <td>NaN</td>
      <td>1746.34</td>
      <td>1.80</td>
      <td>4.07</td>
      <td>2277.13</td>
      <td>...</td>
      <td>NaN</td>
      <td>139.59</td>
      <td>970.47</td>
      <td>1.61</td>
      <td>162.10</td>
    </tr>
    <tr>
      <th>01-JUL-15 08.00.00.000000 AM</th>
      <td>2.07</td>
      <td>6530.64</td>
      <td>6.92</td>
      <td>18.46</td>
      <td>6485.98</td>
      <td>...</td>
      <td>NaN</td>
      <td>334.05</td>
      <td>3064.53</td>
      <td>6.76</td>
      <td>721.93</td>
    </tr>
    <tr>
      <th>01-JUL-15 10.00.00.000000 PM</th>
      <td>1.80</td>
      <td>5853.14</td>
      <td>5.19</td>
      <td>14.20</td>
      <td>4787.97</td>
      <td>...</td>
      <td>NaN</td>
      <td>259.02</td>
      <td>2370.28</td>
      <td>5.70</td>
      <td>606.48</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




    print (df.columns)

    Index([u'Background CPU(s)', u'Block changes', u'DB CPU(s)', u'DB Time(s)',
           u'Executes (SQL)', u'Global Cache blocks received',
           u'Global Cache blocks served', u'Hard parses (SQL)', u'IM scan rows',
           u'Logical read (blocks)', u'Logons', u'Parses (SQL)',
           u'Physical read (blocks)', u'Physical write (blocks)', u'Read IO (MB)',
           u'Read IO requests', u'Redo size (bytes)', u'Rollbacks',
           u'SQL Work Area (MB)', u'Session Logical Read IM', u'Transactions',
           u'User calls', u'Write IO (MB)', u'Write IO requests'],
          dtype='object', name=u'stat_name')


## Distribution Selection

All of these database statistics are continous random variables, so we model using some continous distributions.
scipy offers several continous distributions for the purpose of modelling, viz Gamma, Normal, Anglit, Arcsine, Beta, Cauchy etc.

Once we have the estimated model, we can then use the Kolmogorov-Smirov test to evaluate your fit using scipy.stats.kstest. Lets start with the most popular gamma distribution.

Say I am interested in Background CPU(s) information, calculating the gamma distribution parameters alpha and beta.



    sample_mean=df['Background CPU(s)'].mean()
    sample_var=np.var(df['Background CPU(s)'], ddof=1)
    (sample_mean,sample_var)




    (2.157353689567431, 0.23047410162538273)



Alphas and Betas of Gamma distribution:


    ###############
    #### Gamma ####
    ###############
    
    alpha = sample_mean**2 / sample_var
    beta = sample_mean / sample_var 
    alpha,beta




    (20.19391727342622, 9.360503737092497)



## MOM Estimation

Using estimated parameters and plot the distribution on top of data


    gamma_rv = scs.gamma(a=alpha, scale=1/beta)


    # Get the probability for each value in the data
    x_vals = np.linspace(df['Background CPU(s)'].min(), df['Background CPU(s)'].max())
    gamma_p = gamma_rv.pdf(x_vals)


    df.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>stat_name</th>
      <th>Background CPU(s)</th>
      <th>Block changes</th>
      <th>DB CPU(s)</th>
      <th>DB Time(s)</th>
      <th>Executes (SQL)</th>
      <th>...</th>
      <th>Session Logical Read IM</th>
      <th>Transactions</th>
      <th>User calls</th>
      <th>Write IO (MB)</th>
      <th>Write IO requests</th>
    </tr>
    <tr>
      <th>start_time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>01-APR-15 08.00.00.000000 AM</th>
      <td>2.34</td>
      <td>7171.66</td>
      <td>8.34</td>
      <td>21.33</td>
      <td>7295.85</td>
      <td>...</td>
      <td>NaN</td>
      <td>342.13</td>
      <td>3172.36</td>
      <td>7.38</td>
      <td>791.23</td>
    </tr>
    <tr>
      <th>01-APR-15 10.00.00.000000 PM</th>
      <td>1.88</td>
      <td>5829.02</td>
      <td>5.75</td>
      <td>14.98</td>
      <td>4943.24</td>
      <td>...</td>
      <td>NaN</td>
      <td>243.01</td>
      <td>2288.04</td>
      <td>5.59</td>
      <td>603.88</td>
    </tr>
    <tr>
      <th>01-JAN-15 08.00.00.000000 AM</th>
      <td>NaN</td>
      <td>1746.34</td>
      <td>1.80</td>
      <td>4.07</td>
      <td>2277.13</td>
      <td>...</td>
      <td>NaN</td>
      <td>139.59</td>
      <td>970.47</td>
      <td>1.61</td>
      <td>162.10</td>
    </tr>
    <tr>
      <th>01-JUL-15 08.00.00.000000 AM</th>
      <td>2.07</td>
      <td>6530.64</td>
      <td>6.92</td>
      <td>18.46</td>
      <td>6485.98</td>
      <td>...</td>
      <td>NaN</td>
      <td>334.05</td>
      <td>3064.53</td>
      <td>6.76</td>
      <td>721.93</td>
    </tr>
    <tr>
      <th>01-JUL-15 10.00.00.000000 PM</th>
      <td>1.80</td>
      <td>5853.14</td>
      <td>5.19</td>
      <td>14.20</td>
      <td>4787.97</td>
      <td>...</td>
      <td>NaN</td>
      <td>259.02</td>
      <td>2370.28</td>
      <td>5.70</td>
      <td>606.48</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




    # Generate random colormap
    def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
        """
        Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
        :param nlabels: Number of labels (size of colormap)
        :param type: 'bright' for strong colors, 'soft' for pastel colors
        :param first_color_black: Option to use first color as black, True or False
        :param last_color_black: Option to use last color as black, True or False
        :param verbose: Prints the number of labels and shows the colormap. True or False
        :return: colormap for matplotlib
        """
        from matplotlib.colors import LinearSegmentedColormap
        import colorsys
        import numpy as np
    
        if type not in ('bright', 'soft'):
            print ('Please choose "bright" or "soft" for type')
            return
    
        if verbose:
            print('Number of labels: ' + str(nlabels))
    
        # Generate color map for bright colors, based on hsv
        if type == 'bright':
            randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                              np.random.uniform(low=0.2, high=1),
                              np.random.uniform(low=0.9, high=1)) for i in xrange(nlabels)]
    
            # Convert HSV list to RGB
            randRGBcolors = []
            for HSVcolor in randHSVcolors:
                randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))
    
            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]
    
            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]
    
            random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
    
        # Generate soft pastel colors, by limiting the RGB spectrum
        if type == 'soft':
            low = 0.6
            high = 0.95
            randRGBcolors = [(np.random.uniform(low=low, high=high),
                              np.random.uniform(low=low, high=high),
                              np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]
    
            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]
    
            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]
            random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
    
        # Display colorbar
        if verbose:
            from matplotlib import colors, colorbar
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))
    
            bounds = np.linspace(0, nlabels, nlabels + 1)
            norm = colors.BoundaryNorm(bounds, nlabels)
    
            cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                       boundaries=bounds, format='%1i', orientation=u'horizontal')
    
        return random_colormap


    bright_cmap = rand_cmap(50, type='bright', first_color_black=False, last_color_black=True, verbose=True)

    Number of labels: 50



![png](method_of_moments_files/method_of_moments_23_1.png)



    soft_cmap = rand_cmap(50, type='soft', first_color_black=False, last_color_black=True, verbose=True)

    Number of labels: 50



![png](method_of_moments_files/method_of_moments_24_1.png)



    '''Use the estimated parameters and plot the distribution on top of data'''
    fig, ax = plt.subplots()
    scol=soft_cmap(random.choice(range(50)))
    bcol1=bright_cmap(random.choice(range(50)))
    bcol2=bright_cmap(random.choice(range(50)))
    # Plot those values on top of the real data
    ax = df['Background CPU(s)'].hist(bins=20, normed=1, color=bcol1, figsize=(10, 7))
    ax.set_xlabel('Statistics')
    ax.set_ylabel('Probability Density')
    ax.set_title('Background CPU(s)')
    scol2=bright_cmap(random.choice(range(50)))
    ax.plot(x_vals,  gamma_p, c=bcol2, label='Gamma')
    ax.set_axis_bgcolor(scol)
    ax.legend()
    #py.iplot_mpl(fig, filename='s6_log-scales')




    <matplotlib.legend.Legend at 0x11184a3d0>




![png](method_of_moments_files/method_of_moments_25_1.png)



    # Lets put everything in a function 
    # Define a function that plots distribution fitted to one month's of data
    def plot_mom(df, col):
        sample_mean=df[col].mean()
        sample_var=np.var(df[col], ddof=1)
        alpha = sample_mean**2 / sample_var
        beta = sample_mean / sample_var 
        gamma_rv = scs.gamma(a=alpha, scale=1/beta)
        # Get the probability for each value in the data
        x_vals = np.linspace(df[col].min(), df[col].max())
        gamma_p = gamma_rv.pdf(x_vals)
        fig, ax = plt.subplots()
        # Plot those values on top of the real data
        scol=soft_cmap(random.choice(range(50)))
        bcol1=bright_cmap(random.choice(range(50)))
        bcol2=bright_cmap(random.choice(range(50)))
        ax = df[col].hist(bins=20, normed=1, color=bcol1, figsize=(10, 7))
        ax.set_axis_bgcolor(scol)
        ax.set_xlabel('Statistics')
        ax.set_ylabel('Probability Density')
        ax.set_title(col)
        ax.plot(x_vals, gamma_p, color=bcol2, label='Predicted Values')
        ax.legend()


    df.columns




    Index([u'Background CPU(s)', u'Block changes', u'DB CPU(s)', u'DB Time(s)',
           u'Executes (SQL)', u'Global Cache blocks received',
           u'Global Cache blocks served', u'Hard parses (SQL)', u'IM scan rows',
           u'Logical read (blocks)', u'Logons', u'Parses (SQL)',
           u'Physical read (blocks)', u'Physical write (blocks)', u'Read IO (MB)',
           u'Read IO requests', u'Redo size (bytes)', u'Rollbacks',
           u'SQL Work Area (MB)', u'Session Logical Read IM', u'Transactions',
           u'User calls', u'Write IO (MB)', u'Write IO requests'],
          dtype='object', name=u'stat_name')



Lets just pick the following important statistics to estimate:
	* 'Background CPU(s)'
	* 'DB CPU(s)',
	* 'Executes (SQL)',
	* 'IM scan rows',
	* 'Hard parses (SQL)',
	* 'Logical read (blocks)',
	* 'Parses (SQL)',
	* 'Physical read (blocks)',
	* 'Physical write (blocks)'


    columns_of_interest=['Background CPU(s)', 'DB CPU(s)', 'Executes (SQL)', 'User calls', 'Hard parses (SQL)', 'Logical read (blocks)', 'Parses (SQL)', 'Physical read (blocks)', 'Physical write (blocks)']


    for col in columns_of_interest:
        plot_mom(df, col)
        fig.savefig('images/'+col)


![png](method_of_moments_files/method_of_moments_30_0.png)



![png](method_of_moments_files/method_of_moments_30_1.png)



![png](method_of_moments_files/method_of_moments_30_2.png)



![png](method_of_moments_files/method_of_moments_30_3.png)



![png](method_of_moments_files/method_of_moments_30_4.png)



![png](method_of_moments_files/method_of_moments_30_5.png)



![png](method_of_moments_files/method_of_moments_30_6.png)



![png](method_of_moments_files/method_of_moments_30_7.png)



![png](method_of_moments_files/method_of_moments_30_8.png)



    df.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>stat_name</th>
      <th>Background CPU(s)</th>
      <th>Block changes</th>
      <th>DB CPU(s)</th>
      <th>DB Time(s)</th>
      <th>Executes (SQL)</th>
      <th>...</th>
      <th>Session Logical Read IM</th>
      <th>Transactions</th>
      <th>User calls</th>
      <th>Write IO (MB)</th>
      <th>Write IO requests</th>
    </tr>
    <tr>
      <th>start_time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>01-APR-15 08.00.00.000000 AM</th>
      <td>2.34</td>
      <td>7171.66</td>
      <td>8.34</td>
      <td>21.33</td>
      <td>7295.85</td>
      <td>...</td>
      <td>NaN</td>
      <td>342.13</td>
      <td>3172.36</td>
      <td>7.38</td>
      <td>791.23</td>
    </tr>
    <tr>
      <th>01-APR-15 10.00.00.000000 PM</th>
      <td>1.88</td>
      <td>5829.02</td>
      <td>5.75</td>
      <td>14.98</td>
      <td>4943.24</td>
      <td>...</td>
      <td>NaN</td>
      <td>243.01</td>
      <td>2288.04</td>
      <td>5.59</td>
      <td>603.88</td>
    </tr>
    <tr>
      <th>01-JAN-15 08.00.00.000000 AM</th>
      <td>NaN</td>
      <td>1746.34</td>
      <td>1.80</td>
      <td>4.07</td>
      <td>2277.13</td>
      <td>...</td>
      <td>NaN</td>
      <td>139.59</td>
      <td>970.47</td>
      <td>1.61</td>
      <td>162.10</td>
    </tr>
    <tr>
      <th>01-JUL-15 08.00.00.000000 AM</th>
      <td>2.07</td>
      <td>6530.64</td>
      <td>6.92</td>
      <td>18.46</td>
      <td>6485.98</td>
      <td>...</td>
      <td>NaN</td>
      <td>334.05</td>
      <td>3064.53</td>
      <td>6.76</td>
      <td>721.93</td>
    </tr>
    <tr>
      <th>01-JUL-15 10.00.00.000000 PM</th>
      <td>1.80</td>
      <td>5853.14</td>
      <td>5.19</td>
      <td>14.20</td>
      <td>4787.97</td>
      <td>...</td>
      <td>NaN</td>
      <td>259.02</td>
      <td>2370.28</td>
      <td>5.70</td>
      <td>606.48</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



This above function would generate the MOM estimate for any database statistic that we would want.
For example if we want to predict MOM's for "Global Cache blocks served", we just need to call this function:


    plot_mom(df,'Global Cache blocks received')
    fig.savefig('images/cache_blocks.png')


![png](method_of_moments_files/method_of_moments_33_0.png)


For "Rollbacks" example our estimation is:


    plot_mom(df,'Rollbacks')
    fig.savefig('images/rollbacks.png')


![png](method_of_moments_files/method_of_moments_35_0.png)


## Conclusion:

As we can we our estimations (plotted in red) are pretty good estimation of our actual values that we observe in our real outputs. This concludes our discussion on MOM estimators on database statistics.
