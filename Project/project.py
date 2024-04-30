import pandas as pd
import numpy as np
import stan
from datetime import datetime
from scipy.interpolate import BSpline
from scipy.stats import t
import arviz as az

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import nest_asyncio
nest_asyncio.apply()

# function to determine the season based on the month
def get_season(month):
    if month in ["03", "04", "05"]:
        return 1
    elif month in ["06", "07", "08"]:
        return 2
    elif month in ["09", "10", "11"]:
        return 3
    else:
        return 0

# function to calculate Mother's Day for a given year
def calculate_mothers_day(year):
    first_sunday = pd.Timestamp(f'{year}-05-01').to_pydatetime()
    while first_sunday.weekday() != 6:
        first_sunday += pd.Timedelta(days=1)
    
    second_sunday = first_sunday + pd.Timedelta(days=7)
    
    return second_sunday.strftime("%m/%d/%y")

if __name__ == '__main__':
    
    # read the data
    df = pd.read_csv('BPD_Part_1_Victim_Based_Crime_Data.csv')
    
    # filter for gun violence incidents
    df = df[(df.Description == 'SHOOTING')+((df.Description == 'HOMICIDE')*(df.Weapon == 'FIREARM'))]
    
    # aggregate by date
    df = df.groupby('CrimeDate').size().reset_index(name='shootings')
    df['CrimeDate'] = pd.to_datetime(df.CrimeDate).dt.strftime("%m/%d/%y")
    
    # fill missing dates with zeros
    full_date_range = pd.date_range(start=df['CrimeDate'].min(), end=df['CrimeDate'].max(), freq='D').strftime("%m/%d/%y")
    df = pd.merge(pd.DataFrame({'CrimeDate': full_date_range}), df, on='CrimeDate',how='left')
    df = df.fillna(0)
    
    # define ceasefire weekends
    ceasefire_fridays = [
    "08/04/2017",
    "11/03/2017",
    "02/02/2018",
    "05/11/2018",
    "08/03/2018",
    "11/02/2018",
    "02/01/2019",
    "05/10/2019",
    "08/02/2019"
    ]

    ceasefire_fridays = [datetime.strptime(date_str, "%m/%d/%Y") for date_str in ceasefire_fridays]

    # generate the ceasefire weekends
    ceasefire_weekends = [pd.date_range(start=friday, periods=3, freq='D') for friday in ceasefire_fridays]
    ceasefire_weekends = [date.strftime("%m/%d/%y") for sublist in ceasefire_weekends for date in sublist]
    
    # set binary indicators for ceasefire and related weekends
    df['ceasefire'] = df['CrimeDate'].isin(ceasefire_weekends)
    df['three_after'] = np.roll(df.ceasefire,3)
    df['weekend_after'] = np.roll(df.ceasefire,7)
    
    # define Mother's Day for each year
    mothers_days = []
    for year in pd.to_datetime(df['CrimeDate']).dt.year.unique():
        mothers_days.append(calculate_mothers_day(year))

    df['mothersday'] = df['CrimeDate'].isin(mothers_days)
    
    # encode relevant date information
    day_to_num = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
    df.loc[:,'weekday'] = [day_to_num[x] for x in pd.to_datetime(df.CrimeDate).dt.day_name()]
    df.loc[:,'yearday'] = pd.to_datetime(df.CrimeDate).dt.dayofyear
    df.loc[:,'season'] = [get_season(c.split("/")[0]) for c in df.CrimeDate]
    df.loc[:,'jul'] = pd.to_datetime(df.CrimeDate).apply(lambda x: x.toordinal())

    # fit spline for overall time trend
    knots = np.linspace(df.jul.min(), df.jul.max(), 14)
    degree = 3
    spline_basis = BSpline(knots, np.eye(len(knots)), degree)
    spline_coefficients = np.linalg.lstsq(spline_basis(df.jul), df.shootings, rcond=None)[0]
    eval = df.jul
    overall_spline = np.dot(spline_basis(eval), spline_coefficients)

    # plot overall time trend
    plt.figure(figsize=(12, 6))
    plt.plot(df.CrimeDate, overall_spline, 'r-', label='Spline Trend')
    plt.plot(df.CrimeDate, df.shootings, 'bo', label='Shooting Incidents')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('Overall Time Trend in Shootings')
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(month=12))
    plt.gca().set_xticklabels(['2012','2013','2014','2015','2016','2017','2018','2019'])
    plt.grid(True)
    plt.show()
    
    # fit spline for yearly trend
    year_counts = np.zeros(366)
    for yearday, count in zip(df.yearday, df.shootings):
        year_counts[yearday-1] += count    
    year_counts = year_counts/(len(df)/365)
    
    knots = np.linspace(0, 366, 12)
    degree = 2
    spline_basis = BSpline(knots, np.eye(len(knots)), degree, extrapolate='periodic')

    spline_coefficients = np.linalg.lstsq(spline_basis(np.arange(366)), year_counts, rcond=None)[0]

    eval = np.arange(0, 366)
    year_spline = np.dot(spline_basis(eval), spline_coefficients)

    # plot yearly trend
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(366), year_counts, 'bo', label='Shooting Incidents')
    plt.plot(eval, year_spline, 'r-', label='Spline Trend')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('Yearly trend in Shootings')
    plt.legend()
    plt.grid(True)
    plt.show()

    N = len(df)
    
    # prepare data for Stan model
    data = {
        'n': N,
        'y': df['shootings'].values.astype(int),
        'overall': (df['jul']-df['jul'][0]).values.astype(int)+1,
        'ceasefire': df['ceasefire'].values.astype(int)+1,
        'residual': df['three_after'].values.astype(int)+1,
        'residual_weekend': df['weekend_after'].values.astype(int)+1,
        'mothersday': df['mothersday'].values.astype(int)+1,
        'yday': df['yearday'].values.astype(int),
        'weekday': df['weekday'].values.astype(int)+1,
        'season': df['season'].values.astype(int)+1,
        'y_overall': overall_spline,
        'y_year': year_spline
    }

    stan_code = """
    data {
        int<lower=1> n;                   // Number of observations
        array[n] int y;               // Number of shootings per day
        array[n] int overall;         //overall time period
        array[n] int ceasefire;
        array[n] int residual;
        array[n] int residual_weekend;
        array[n] int mothersday;
        array[n] int yday;
        array[n] int weekday;
        array[n] int season;
        
        vector[n] y_overall;                //overall spline
        vector[366] y_year;                 //yearly spline
        
    }

    parameters {
        vector[n] mu_innovations;
        vector[7] weekday_effects;
        vector[2] ceasefire_effects;
        vector[2] residual_effects;
        vector[2] residual_weekend_effects;
        vector[2] mothersday_effects;
        vector[4] season_effects;
        
        real<lower=0> sigma_mu;
        real<lower=0> sigma_weekday;
        real<lower=0> sigma_binary;
        real<lower=0> sigma_seasonal;
        real baseline;
        }

    transformed parameters {
        vector[7] y_week;
        vector[2] y_cease;
        vector[2] y_resid;
        vector[2] y_resid_weekend;
        vector[2] y_mother;
        vector[4] y_seasonal;
        vector[n] mu;
        {
            vector[7] week_with_trend;
            real trend;
            week_with_trend = cumulative_sum(weekday_effects);
            trend = week_with_trend[7];
            for (i in 1:7)
                y_week[i] = sigma_weekday/100 * (week_with_trend[i] - trend * i/7);
        }
        {
            vector[2] cease_with_trend;
            vector[2] mother_with_trend;
            vector[2] resid_with_trend;
            vector[2] resid_weekend_trend;
            real cease_trend;
            real mother_trend;
            real resid_trend;
            real resid_weekend;
            cease_with_trend = cumulative_sum(ceasefire_effects);
            mother_with_trend = cumulative_sum(mothersday_effects);
            resid_with_trend = cumulative_sum(residual_effects);
            resid_weekend_trend = cumulative_sum(residual_weekend_effects);
            cease_trend = cease_with_trend[2];
            mother_trend = mother_with_trend[2];
            resid_trend = resid_with_trend[2];
            resid_weekend = resid_weekend_trend[2];
            for (i in 1:2) {
                y_cease[i] = sigma_binary/100 * (cease_with_trend[i] - cease_trend * i/2);
                y_mother[i] = sigma_binary/100 * (mother_with_trend[i] - mother_trend * i/2);
                y_resid[i] = sigma_binary/100 * (resid_with_trend[i] - resid_trend * i/2);
                y_resid_weekend[i] = sigma_binary/100 * (resid_weekend_trend[i] - resid_weekend * i/2);
            }
        }
        {
            vector[4] season_with_trend;
            real trend;
            season_with_trend = cumulative_sum(season_effects);
            trend = season_with_trend[4];
            for (i in 1:4)
                y_seasonal[i] = sigma_seasonal/100 * (season_with_trend[i] - trend * i/4);
        }
        mu = sigma_mu/100 * cumulative_sum(mu_innovations);
    }

    model {
        weekday_effects ~ normal(0,1);
        ceasefire_effects ~ normal(0,1);
        residual_effects ~ normal(0,1);
        residual_weekend_effects ~ normal(0,1);
        mothersday_effects ~ normal(0,1);
        season_effects ~ normal(0,1);
        mu_innovations ~ normal(0, 1);
        y ~ poisson_log(baseline + mu + y_year[yday] + (y_overall[overall]) + y_week[weekday] + y_cease[ceasefire] + y_resid[residual] + y_resid_weekend[residual_weekend] + y_mother[mothersday] + y_seasonal[season]);
        sigma_mu ~ lognormal(-3.5 + log(100), 2);
        sigma_weekday ~ lognormal(-3.5 + log(100), 2);
        sigma_binary ~ lognormal(-3.5 + log(100), 2);
        sigma_seasonal ~ lognormal(-3.5 + log(100), 2);
    }

    generated quantities {
        vector[n] lin_pred;    // Linear predictor
        vector[n] lower_bound;
        vector[n] upper_bound;


        // Compute the linear predictor for each observation
        for (i in 1:n) {
            lin_pred[i] = baseline + mu[i] + y_year[yday[i]] + (2*y_overall[overall[i]]) + y_week[weekday[i]] + y_cease[ceasefire[i]] + y_resid[residual[i]] + y_resid_weekend[residual_weekend[i]] + y_mother[mothersday[i]] + y_seasonal[season[i]];
            real interval_width = 1.645 * sqrt(square(sigma_mu) + square(sigma_weekday/100) + square(sigma_binary/100) + square(sigma_seasonal/100));
            lower_bound[i] = lin_pred[i] - interval_width / 2;
            upper_bound[i] = lin_pred[i] + interval_width / 2;
        }
        
    }

    """

    # build and sample the model
    model = stan.build(stan_code, data=data)
    fit = model.sample(num_chains=4, num_samples=500, num_warmup=500)

    # summary and trace plot
    '''print(az.summary(fit))
    _ = az.plot_trace(fit)
    
    corrs = az.autocorr(fit['mu_innovations'])
    
    corrs = corrs.reshape(corrs.shape[0],500,4)
    for chain in range(corrs.shape[0]):
        plt.plot(corrs[chain,:, 3])
    plt.title('Chain 4')'''

    # plot model vs observations
    plt.figure(figsize=(10, 6))
    plt.scatter(df['CrimeDate'], df['shootings'], alpha=0.2, label='Observations')
    plt.plot(df['CrimeDate'], np.mean(fit['lin_pred'], axis=1), color='red', alpha=0.5, label='Model Prediction')
    plt.fill_between(df['CrimeDate'], np.mean(fit['hdi_3%'], axis=1), np.mean(fit['hdi_97%'], axis=1), color='blue', alpha=0.2)
    plt.ylim(0,12)
    plt.xlabel('Date')
    plt.ylabel('Daily Shootings')
    plt.title('Model vs Observations')
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(month=12))
    plt.gca().set_xticklabels(['2012','2013','2014','2015','2016','2017','2018','2019'])
    plt.grid(True)
    plt.show()
    
    # plot weekday effects
    plt.figure(figsize=(10, 6))
    plt.boxplot(fit.to_frame().loc[:,["weekday_effects.2","weekday_effects.3","weekday_effects.4","weekday_effects.5","weekday_effects.6","weekday_effects.7","weekday_effects.1"]])
    plt.gca().set_xticklabels(['Monday', 'Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    plt.ylabel('Daily Shootings')
    plt.title('Weekday effects')
    plt.show()
    
    results = fit.to_frame()
    
    # calculate and display credible interval for Mother's Days
    mother_effect_samples = results["mothersday_effects.2"]
    point_estimate = np.mean(mother_effect_samples)
    credible_interval = np.percentile(mother_effect_samples, [2.5, 97.5])

    print("Point Estimate for Shootings on Mothers' day:", point_estimate)
    print("95% Credible Interval:", credible_interval)
    
    # calculate and display credible interval for ceasfire weekends
    ceasefire_effect_samples = results["ceasefire_effects.2"]
    point_estimate = np.mean(ceasefire_effect_samples)
    credible_interval = np.percentile(ceasefire_effect_samples, [2.5, 97.5])

    print("Point Estimate for Shootings on Ceasefire Days:", point_estimate)
    print("95% Credible Interval:", credible_interval)
    
    # calculate and display credible interval for the days following the ceasefire weekends
    resid_effect_samples = results["residual_effects.2"]
    point_estimate = np.mean(resid_effect_samples)
    credible_interval = np.percentile(resid_effect_samples, [2.5, 97.5])

    print("Point Estimate for Shootings on following days:", point_estimate)
    print("95% Credible Interval:", credible_interval)
    
    # calculate and display credible interval for weekend after ceasfire weekends
    next_effect_samples = results["residual_weekend_effects.2"]
    point_estimate = np.mean(next_effect_samples)
    credible_interval = np.percentile(next_effect_samples, [2.5, 97.5])

    print("Point Estimate for Shootings on following weekends:", point_estimate)
    print("95% Credible Interval:", credible_interval)