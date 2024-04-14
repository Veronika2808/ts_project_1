import numpy as np
import pandas as pd
def alarm_changepoint(df, alpha=0.1, h_high = 0.3, h_low = -0.4):
    """
    Find changepoints in data by mean comparison
    In:
        df: data
        alpha: new_data weight
        h_high: full_df.target.mean() + full_df.target.std()
        h_low: full_df.target.mean() - full_df.target.std()
    Out:
        changepoint indicator, date of changepoint if present 

    """
    dates = list(sorted(df['date'].unique()))
    stats = []
    for i in range(1, len(dates)):
        current_value = df[df['date']==dates[i]]['target'].values[0]
        prev_values = list(df[df['date']<=dates[i]]['target'].values)
        means = np.mean(prev_values)
        stat = (1-alpha)*means + alpha*current_value
        stats.append(stat)
        if stat >= h_high or stat<= h_low:
            print('changepoint, shift in data')
    
    breakpoint1 = None
    breakpoint2 = None
    if len(list(np.where(np.array(stats) >= h_high)[0]))>0:
        breakpoint1= list(np.where(np.array(stats) >= h_high)[0])[0]

    if len(list(np.where(np.array(stats) <=  h_low)[0]))>0:
        breakpoint2= list(np.where(np.array(stats) <=  h_low)[0])[0]
    if breakpoint1 is not None and breakpoint2 is not None:
        return True, dates[min(breakpoint1, breakpoint2)]
    elif breakpoint1 is not None:
        return True, dates[breakpoint1]
    elif breakpoint2 is not None:
        return  True, dates[breakpoint2]
    else:
        print('No breakpoint')
        return False, 0