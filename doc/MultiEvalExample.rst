
For example:

.. code:: ipython3

    from lenskit.batch import MultiEval
    from lenskit.crossfold import partition_users, SampleN
    from lenskit.algorithms import basic, als
    from lenskit.util import load_ml_ratings
    from lenskit import topn
    import pandas as pd

Generate the train-test pairs:

.. code:: ipython3

    pairs = list(partition_users(load_ml_ratings(), 5, SampleN(5)))

Set up and run the ``MultiEval`` experiment:

.. code:: ipython3

    eval = MultiEval('my-eval', recommend=20)
    eval.add_datasets(pairs, name='ML-Small')
    eval.add_algorithms(basic.Popular(), name='Pop')
    eval.add_algorithms([als.BiasedMF(f) for f in [20, 30, 40, 50]],
                        attrs=['features'], name='ALS')
    eval.run()

Now that the experiment is run, we can read its outputs.

First the run metadata:

.. code:: ipython3

    runs = pd.read_csv('my-eval/runs.csv')
    runs.set_index('RunId', inplace=True)
    runs.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>AlgoClass</th>
          <th>AlgoStr</th>
          <th>DataSet</th>
          <th>Partition</th>
          <th>PredTime</th>
          <th>RecTime</th>
          <th>TrainTime</th>
          <th>features</th>
          <th>name</th>
        </tr>
        <tr>
          <th>RunId</th>
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
          <th>1</th>
          <td>Popular</td>
          <td>Popular</td>
          <td>ML-Small</td>
          <td>1</td>
          <td>NaN</td>
          <td>0.578916</td>
          <td>0.278333</td>
          <td>NaN</td>
          <td>Pop</td>
        </tr>
        <tr>
          <th>2</th>
          <td>BiasedMF</td>
          <td>als.BiasedMF(features=20, regularization=0.1)</td>
          <td>ML-Small</td>
          <td>1</td>
          <td>0.377277</td>
          <td>1.324478</td>
          <td>5.426510</td>
          <td>20.0</td>
          <td>ALS</td>
        </tr>
        <tr>
          <th>3</th>
          <td>BiasedMF</td>
          <td>als.BiasedMF(features=30, regularization=0.1)</td>
          <td>ML-Small</td>
          <td>1</td>
          <td>0.326613</td>
          <td>1.566073</td>
          <td>1.300490</td>
          <td>30.0</td>
          <td>ALS</td>
        </tr>
        <tr>
          <th>4</th>
          <td>BiasedMF</td>
          <td>als.BiasedMF(features=40, regularization=0.1)</td>
          <td>ML-Small</td>
          <td>1</td>
          <td>0.408973</td>
          <td>1.570634</td>
          <td>1.904973</td>
          <td>40.0</td>
          <td>ALS</td>
        </tr>
        <tr>
          <th>5</th>
          <td>BiasedMF</td>
          <td>als.BiasedMF(features=50, regularization=0.1)</td>
          <td>ML-Small</td>
          <td>1</td>
          <td>0.357133</td>
          <td>1.700047</td>
          <td>2.390314</td>
          <td>50.0</td>
          <td>ALS</td>
        </tr>
      </tbody>
    </table>
    </div>



Then the recommendations:

.. code:: ipython3

    recs = pd.read_parquet('my-eval/recommendations.parquet')
    recs.head()


.. parsed-literal::

    D:\Anaconda3\lib\site-packages\pyarrow\pandas_compat.py:698: FutureWarning: .labels was deprecated in version 0.24.0. Use .codes instead.
      labels = getattr(columns, 'labels', None) or [
    D:\Anaconda3\lib\site-packages\pyarrow\pandas_compat.py:725: FutureWarning: the 'labels' keyword is deprecated, use 'codes' instead
      return pd.MultiIndex(levels=new_levels, labels=labels, names=columns.names)
    D:\Anaconda3\lib\site-packages\pyarrow\pandas_compat.py:742: FutureWarning: .labels was deprecated in version 0.24.0. Use .codes instead.
      labels, = index.labels
    



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>item</th>
          <th>score</th>
          <th>user</th>
          <th>rank</th>
          <th>RunId</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>356</td>
          <td>335</td>
          <td>6</td>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>296</td>
          <td>323</td>
          <td>6</td>
          <td>2</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>318</td>
          <td>305</td>
          <td>6</td>
          <td>3</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>593</td>
          <td>302</td>
          <td>6</td>
          <td>4</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>260</td>
          <td>284</td>
          <td>6</td>
          <td>5</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



In order to evaluate the recommendation list, we need to build a
combined set of truth data. Since this is a disjoint partition of users
over a single data set, we can just concatenate the individual test
frames:

.. code:: ipython3

    truth = pd.concat((p.test for p in pairs), ignore_index=True)

Now we can set up an analysis and compute the results.

.. code:: ipython3

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    ndcg = rla.compute(recs, truth)
    ndcg.head()

Next, we need to combine this with our run data, so that we know what
algorithms and configurations we are evaluating:

.. code:: ipython3

    ndcg = ndcg.join(runs[['AlgoClass', 'features']], on='RunId')
    ndcg.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>ndcg</th>
          <th>AlgoClass</th>
          <th>features</th>
        </tr>
        <tr>
          <th>user</th>
          <th>RunId</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="5" valign="top">1</th>
          <th>11</th>
          <td>0.0</td>
          <td>Popular</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.0</td>
          <td>BiasedMF</td>
          <td>20.0</td>
        </tr>
        <tr>
          <th>13</th>
          <td>0.0</td>
          <td>BiasedMF</td>
          <td>30.0</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.0</td>
          <td>BiasedMF</td>
          <td>40.0</td>
        </tr>
        <tr>
          <th>15</th>
          <td>0.0</td>
          <td>BiasedMF</td>
          <td>50.0</td>
        </tr>
      </tbody>
    </table>
    </div>



The Popular algorithm has NaN feature count, which ``groupby`` doesn’t
like; let’s fill those in.

.. code:: ipython3

    ndcg.loc[ndcg['AlgoClass'] == 'Popular', 'features'] = 0

And finally, we can compute the overall average performance for each
algorithm configuration:

.. code:: ipython3

    ndcg.groupby(['AlgoClass', 'features'])['ndcg'].mean()




.. parsed-literal::

    AlgoClass  features
    BiasedMF   20.0        0.015960
               30.0        0.022558
               40.0        0.025901
               50.0        0.028949
    Popular    0.0         0.091814
    Name: ndcg, dtype: float64


