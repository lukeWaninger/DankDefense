import datetime as dt
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

NO_DETECTIONS = '#706C60'
DETECTIONS = '#FFAE09'

dp = lambda x: os.path.join('/', 'mnt', 'f', 'DankDefense', x)


def write_feat(name, train, test):
    path = dp(os.path.join('feats', name))

    train.to_csv(dp(f'{path}_train.csv'), index=None)
    test.to_csv(dp(f'{path}_test.csv'), index=None)


def get_feat(name, t):
    path = dp(os.path.join('feats', name))

    train = pd.read_csv(dp(f'{path}_train.csv'), dtype=t)
    test = pd.read_csv(dp(f'{path}_test.csv'), dtype=t)
    return train, test


def clear(name):
    if not isinstance(name, list):
        name = [name]

    for v in name:
        if v in globals().keys():
            del globals()[v]
            x = gc.collect()


def nanp(df, show_zero=False):
    cols = df.columns
    d, p = len(df), []

    for i, col in enumerate(cols):
        a = sum(pd.isnull(df[col]))
        p.append([col, df[col].dtype, np.round(a / d * 100, 1)])

    p = pd.DataFrame(p, columns=['Variable', 'DataType', 'PercentNA'])

    if not show_zero:
        return p.loc[p['PercentNA'] > 0].sort_values(by='PercentNA', ascending=False)
    else:
        return p.sort_values(by='PercentNA', ascending=False)


def isfloat(x):
    try:
        float(x)
        return True
    except:
        return False


def isint(x):
    try:
        int(x)
        return True
    except:
        return False


def printcats(df, c):
    df[c] = df[c].apply(lambda x: str(x).lower() if not pd.isnull(x) else np.nan)

    df.loc[
        (df.loc[:, c] == 'unknown') |
        (df.loc[:, c] == 'unspecified') |
        df.loc[:, c].isnull(), c
    ] = np.nan

    un = df[c].unique()
    if len(un) < 20:
        print(c, len(c), ':', un)
    else:
        print(c, len(c), ':', ', '.join([str(x) for x in un[:5]]) + ', ...')


def pcols(df):
    t = [print(c) for c in sorted(list(set(df.columns)))]


def cateval(df, c, test_data=False):
    print(f'{"test" if test_data else "train"} percent na: ', df[c].isnull().mean())

    if not test_data:
        t = pd.crosstab(df[c], df.HasDetections, normalize='index').sort_values(c)
        t['total_count'] = df[c].value_counts()
        t['normalized'] = t.total_count / t.total_count.sum()
    else:
        t = pd.value_counts(df[c])
        t['normalized'] = pd.value_counts(df[c], normalize=True)
        t.columns = ['count', 'ratio']
        print(t)


class DFMan(object):
    def __init__(self, dtypes):
        self.dtypes = dtypes

    @property
    def cat_cols(self):
        return sorted([c for c, v in self.dtypes.items() if v == 'category'])

    @property
    def bin_cols(self):
        return sorted([c for c, v in self.dtypes.items() if v == 'int8'])

    @property
    def num_cols(self):
        return sorted(list(set(self.dtypes.keys()) - set(self.cat_cols) - set(self.bin_cols)))

    @property
    def dict(self):
        return self.dtypes

    def add_type(self, k, v):
        self.dtypes[k] = v

    def remove_type(self, k):
        del self.dtypes[k]


def cat_over_time(train, test, c, close=False):
    try:
        if len(train[c].unique()) > 15:
            return

        def fx(df_):
            ct = df_[c].value_counts(normalize=True)
            return ct

        df_train = train[['avsig_dt', 'MachineIdentifier', c]].groupby(['avsig_dt']).apply(fx).reset_index()
        df_test = test[['avsig_dt', 'MachineIdentifier', c]].groupby(['avsig_dt']).apply(fx).reset_index()

        df = pd.concat([df_train, df_test], axis=0, sort=False).sort_values(by='avsig_dt')
        df.columns = ['date', c, 'perc']
        df[c] = df[c].astype(str)

        del df_train, df_test
        x = gc.collect()

        x = plt.gcf()
        x = sns.set(style='whitegrid')

        fig, ax1 = plt.subplots()

        x = fig.set_size_inches(17, 10)
        x = plt.xlim((dt.date(year=2018, month=6, day=1), max(df.date) + dt.timedelta(days=7)))
        x = plt.grid(False)
        x = plt.title(f'{c} over time')

        # for ci in df[c].unique():
        x = sns.lineplot(
            x='date',
            y='perc',
            hue=c,
            data=df
        )

        x = plt.savefig(dp(os.path.join('figs', f'time_category_{c}.png')))

        if close:
            x = plt.close()
    except Exception as e:
        print(f'failed: {c}')


def cat_by_detections(df, c, close=False):
    try:
        n = len(df[c].unique())

        if n > 15:
            return

        df_ = df[[c, 'HasDetections', 'AvSigVersion']]\
            .groupby([c, 'HasDetections'])\
            .count()\
            .reset_index()
        df_['color'] = ''
        df_.loc[df_.HasDetections == 0, 'color'] = NO_DETECTIONS
        df_.loc[df_.HasDetections == 1, 'color'] = DETECTIONS

        x = plt.gcf()
        fig, ax1 = plt.subplots()
        x = fig.set_size_inches(17, 10)

        x = sns.barplot(
            x=c,
            y='AvSigVersion',
            hue='HasDetections',
            palette={0: NO_DETECTIONS, 1: DETECTIONS},
            data=df_
        )

        x = plt.savefig(dp(os.path.join('figs', f'{c}_HasDetections.png')))

        if close:
            x = plt.close()

        del df_, fig
        x = gc.collect()
    except Exception as e:
        print(e)
        print(f'failed {c}')


def numeric_over_time(train, test, c, close=False):
    try:
        train_name = f'mean_{c}_train'
        test_name = f'mean_{c}_test'

        df_train = train[[c, 'avsig_dt', 'HasDetections']]\
            .groupby(['avsig_dt'])\
            .agg([np.mean])\
            .reset_index()
        df_train.columns = ['dt', train_name, 'mean_detections']

        df_test = test[[c, 'avsig_dt']]\
            .groupby(['avsig_dt'])\
            .agg([np.mean])\
            .reset_index()
        df_test.columns = ['dt', test_name]

        df = df_train.merge(df_test, on='dt', how='outer')
        df = df.fillna(0)

        del df_train, df_test
        gc.collect()

        plt.gcf()
        sns.set(style='whitegrid')

        fig, ax1 = plt.subplots()

        x = fig.set_size_inches(17, 10)
        x = plt.xlim((dt.date(year=2018, month=6, day=1), dt.date(year=2018, month=12, day=1)))
        x = plt.grid(False)
        x = ax1.set_title(f'Mean {c}', fontsize=22)
        x = ax1.set_ylabel('Mean AV Products Installed', fontsize=20)

        x = sns.lineplot(
            x='dt',
            y=train_name,
            data=df,
            ax=ax1,
            linewidth=2,
            legend='brief',
            dashes=False
        )
        x = sns.lineplot(
            x='dt',
            y=test_name,
            data=df,
            ax=ax1,
            linewidth=2,
            legend='brief',
            dashes=False
        )

        x = plt.savefig(dp(os.path.join('figs', f'time_numerical_{c}.png')))

        if close:
            x = plt.close()
    except Exception as e:
        print(e)
        print(f'failed {c}')


def numeric_by_detections(df, c, close=False):
    try:
        x = plt.gcf()
        fig, ax1 = plt.subplots()
        x = fig.set_size_inches(13, 6)
        x = plt.grid(False)

        x = sns.distplot(
            df.loc[df.HasDetections == 0, c].sample(20_000),
            hist=False,
            kde=True,
            kde_kws={"shade": True},
            color=NO_DETECTIONS
        )
        x = sns.distplot(
            df.loc[df.HasDetections == 1, c].sample(20_000),
            hist=False,
            kde=True,
            kde_kws={"shade": True},
            color=DETECTIONS
        )

        x = plt.savefig(os.path.join('figs', f'{c}_HasDetections.png'))

        if close:
            x = plt.close()

        del fig
        x = gc.collect()
    except Exception as e:
        print(e)
        print(f'failed {c}')