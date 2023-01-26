import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def pearson_global(flow,traces):
    flow = np.ndarray.flatten(flow)
    traces=np.ndarry.flatten(traces)
    r, p = stats.pearsonr(flow,traces)
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")   

    return r,p

def pearson_local(flow,traces):
    flow = np.ndarray.flatten(flow)
    traces=np.ndarry.flatten(traces)
    data = [{flow},{traces}]
    df=pd.DataFrame(data,columns=['flow', 'traces'])
    r_window_size = 120
    # Interpolate missing data.
    df_interpolated = df.interpolate()
    # Compute rolling window synchrony
    rolling_r = df_interpolated['S1_Joy'].rolling(window=r_window_size, center=True).corr(df_interpolated['S2_Joy'])
    f,ax=plt.subplots(2,1,figsize=(14,6),sharex=True)
    df.rolling(window=30,center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Frame',ylabel='Smiling Evidence')
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Frame',ylabel='Pearson r')
    plt.suptitle("Smiling data and rolling window correlation")


   




