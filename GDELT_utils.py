import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.patches import Polygon
from matplotlib.colorbar import ColorbarBase
from sklearn.model_selection import train_test_split

def weak_reduce(sample, column, pvalue, mag, count_penalty):
    tones = sample['AVG(AvgTone)']
    cntry = sample[column]

    one_hot = pd.get_dummies(cntry)
    one_hot_tone = pd.concat([tones, one_hot], axis=1)

    avg_avgtone_mean = one_hot_tone['AVG(AvgTone)'].mean()

    country_info = []
    for column in one_hot.columns:
        temp = one_hot_tone[[column, 'AVG(AvgTone)']]
        country = temp[temp[column] == 1]
        columns_mean = None
        if len(country) < count_penalty: 
            column_means = [0, 0]
            country_info.append((column, 
                                 0, 
                                 0, 
                                 (temp[column].sum()),
                                 1))
        else:
            column_means = temp.groupby(column).mean()['AVG(AvgTone)']
            country_info.append((column, 
                                 column_means[0] - column_means[1], 
                                 np.absolute(column_means[0] - column_means[1]), 
                                 (temp[column].sum()),
                                 ttest_1samp(country, avg_avgtone_mean).pvalue[1]))

    cntry_spec = pd.DataFrame(country_info, columns=["Country", "AvgTone_diff", "AvgTone_mag", "Num", "p-value"])

    low_decs = cntry_spec[((cntry_spec['p-value'] > pvalue) & ((cntry_spec['AvgTone_mag'] < mag) & \
               (cntry_spec['AvgTone_diff'] < 0)))]['Country']

    low_incs = cntry_spec[((cntry_spec['p-value'] > pvalue) & ((cntry_spec['AvgTone_mag'] < mag) & \
               (cntry_spec['AvgTone_diff'] > 0)))]['Country']

    low_p_value = cntry_spec[((cntry_spec['p-value'] > pvalue) & ~((cntry_spec['AvgTone_mag'] < mag) & \
               (cntry_spec['AvgTone_diff'] < 0)) & ~((cntry_spec['AvgTone_mag'] < mag) & \
               (cntry_spec['AvgTone_diff'] > 0)))]['Country']
    
    return low_decs, low_incs, low_p_value

def map_weak(x, low_decs, low_incs, low_p_value, translation):
    #if x in l:
    #    return x
    if x == "nan":
        return "UNKNOWN"
    elif x in low_decs:
        return "LOW_DEC"
    elif x in low_incs:
        return "LOW_INCS"
    elif x in low_p_value:
        return "LOW_P_VALUE"
    else:
        return x

def unify_weak_cats(data, category, pvalue, mag, count_penalty):
    low_decs, low_incs, low_p_value = weak_reduce(data, category, pvalue, mag)
    low_decs_unique = low_decs.unique()
    low_decs_unique = low_incs.unique()
    low_p_value_unique = low_p_value.unique()
    data[(category + '_unify')] = data[category].astype(str) \
                                                .apply(lambda x: map_missing(x, low_decs_unique, 
                                                                             low_decs_unique, 
                                                                             low_p_value_unique, 
                                                                             "OTHER")) \
                                                .astype('category')
            
def map_rare(x, l, translation):
    if x in l:
        return x
    elif x == "nan":
        return "UNKNOWN"
    else:
        return translation

def unify_rare_cats(data, category, cut_off):
    vc = data[category].value_counts()
    past_cut_off = (vc/len(data)) > cut_off
    remaining = list(vc[past_cut_off].index)
    data[(category + '_unify')] = data[category].astype(str) \
                                                .apply(lambda x: map_missing(x, remaining, "OTHER")) \
                                                .astype('category')
            
def map_unknown(x):
    if x == 'nan':
        return "UNKNOWN"
    else:
        return x            

def init_sample(gdelt, frac): 
    gdelt_sample = gdelt.sample(frac=frac) \
                        .drop(['SQLDATE'], axis=1)
    gdelt_sample['norm_NumMentions'] = (gdelt_sample['AVG(NumMentions)'] \
                                                - gdelt_sample['AVG(NumMentions)'].mean())/ \
                                        gdelt_sample['AVG(NumMentions)'].std()
    gdelt_sample = gdelt_sample.drop(['AVG(NumMentions)'], axis=1) 
    for category_col in gdelt_sample.columns:
        if hasattr(gdelt_sample[category_col], 'cat'):
            gdelt_sample[category_col] = gdelt_sample[category_col] \
                                            .cat.remove_unused_categories()
    return gdelt_sample.copy()    

def pare(sample):
    return sample.drop(['Actor1Geo_CountryCode', 'Actor2Geo_CountryCode'], axis=1)

naive_cache = {}

# https://github.com/pandas-dev/pandas/issues/8814
def train_naive(sample, model):
    h = pd.util.hash_pandas_object(sample).sum()
    model_samp, feat_cols = None, None
    if h not in naive_cache:
        live_samp = sample.copy()

        cat_dummies = []
        drop_cols = []
        for column in live_samp.columns:
            if hasattr(live_samp[column], 'cat'):
                live_samp[column] = live_samp[column].cat.add_categories(['UNK'])
                live_samp[column].fillna('UNK')
                hot = pd.get_dummies(live_samp[column], prefix=column)
                cat_dummies.append(hot)
                drop_cols.append(column)

        live_samp = live_samp.drop(drop_cols, axis=1)            

        one_hot_enc = pd.concat(cat_dummies, axis=1)

        model_samp = pd.concat([live_samp, one_hot_enc], axis=1)
        feat_cols = model_samp.columns.drop(['AVG(AvgTone)'])
        
        naive_cache[h] = (model_samp, feat_cols)
    else:
        model_samp, feat_cols = naive_cache[h]
    
    train, test = train_test_split(model_samp, test_size=0.25, random_state=42)
    
    Y = train['AVG(AvgTone)']
    X = train[feat_cols]
    model.fit(X, Y)

    train_score = model.score(X, Y)
    Y_test = test['AVG(AvgTone)']
    X_test = test[feat_cols]
    test_score = model.score(X_test, Y_test)
    
    return train_score, test_score, model, model_samp, train, test

pared_cache = {}

def train_pared(sample, model):
    h = pd.util.hash_pandas_object(sample).sum()
    model_samp, feat_cols = None, None
    if h not in pared_cache:
        live_samp = sample.copy()
        live_samp = pare(live_samp)

        cat_dummies = []
        drop_cols = []
        for column in live_samp.columns:
            if hasattr(live_samp[column], 'cat'):
                hot = pd.get_dummies(live_samp[column], prefix=column)
                cat_dummies.append(hot)
                drop_cols.append(column)

        live_samp = live_samp.drop(drop_cols, axis=1)            

        one_hot_enc = pd.concat(cat_dummies, axis=1)

        model_samp = pd.concat([live_samp, one_hot_enc], axis=1)
        feat_cols = model_samp.columns.drop(['AVG(AvgTone)'])
        
        pared_cache[h] = (model_samp, feat_cols)
    else:
        model_samp, feat_cols = naive_cache[h]        

    train, test = train_test_split(model_samp, test_size=0.25, random_state=42)
    
    Y = train['AVG(AvgTone)']
    X = train[feat_cols]
    model.fit(X, Y)

    train_score = model.score(X, Y)
    Y_test = test['AVG(AvgTone)']
    X_test = test[feat_cols]
    test_score = model.score(X_test, Y_test)
    
    return train_score, test_score, model, model_samp, train, test

def train_naive_URARE(sample, model, cut_off=.005):
    live_samp = sample.copy()
    
    cat_dummies = []
    drop_cols = []
    for column in live_samp.columns:
        if hasattr(live_samp[column], 'cat'):
            unify_rare_cats(live_samp, column, cut_off)
            hot = pd.get_dummies(live_samp[column], prefix=column)
            cat_dummies.append(hot)
            drop_cols.append(column)
            
    live_samp.drop(drop_cols, axis=1)            
    
    one_hot_enc = pd.concat(cat_dummies, axis=1)
    
    model_samp = pd.concat([live_samp, one_hot_enc])
    feat_cols = model_samp.columns.drop(['AVG(AvgTone)'])
    
    train, test = train_test_split(live_samp, test_size=0.25, random_state=42)
    
    Y = train['AVG(AvgTone)']
    X = train[feat_cols]
    model.fit(X, Y)
    
    return train_score, test_score, model, model_samp, train, test

def train_pared_URARE(sample, model, cut_off=.005):
    live_samp = sample.copy()
    live_samp = pare(live_samp)
    
    cat_dummies = []
    drop_cols = []
    for column in live_samp.columns:
        if hasattr(live_samp[column], 'cat'):
            unify_rare_cats(live_samp, column, cut_off)
            hot = pd.get_dummies(live_samp[column], prefix=column)
            cat_dummies.append(hot)
            drop_cols.append(column)
            
    live_samp.drop(drop_cols, axis=1)            
    
    one_hot_enc = pd.concat(cat_dummies, axis=1)
    
    model_samp = pd.concat([live_samp, one_hot_enc])
    feat_cols = model_samp.columns.drop(['AVG(AvgTone)'])
    
    train, test = train_test_split(live_samp, test_size=0.25, random_state=42)
    
    Y = train['AVG(AvgTone)']
    X = train[feat_cols]
    model.fit(X, Y)
    
    return train_score, test_score, model, model_samp, train, test

def train_naive_UWEAK(sample, model, pvalue=0.0001, mag=1, count_penalty=10):
    live_samp = sample.copy()
    
    cat_dummies = []
    drop_cols = []
    for column in live_samp.columns:
        if hasattr(live_samp[column], 'cat'):
            unify_weak_cats(live_samp, column, pvalue, mag, count_penalty)
            hot = pd.get_dummies(live_samp[column], prefix=column)
            cat_dummies.append(hot)
            drop_cols.append(column)
            
    live_samp.drop(drop_cols, axis=1)            
    
    one_hot_enc = pd.concat(cat_dummies, axis=1)
    
    model_samp = pd.concat([live_samp, one_hot_enc])
    feat_cols = model_samp.columns.drop(['AVG(AvgTone)'])
    
    train, test = train_test_split(live_samp, test_size=0.25, random_state=42)
    
    Y = train['AVG(AvgTone)']
    X = train[feat_cols]
    model.fit(X, Y)
    
    return train_score, test_score, model, model_samp, train, test

def train_pared_UWEAK(sample, model, pvalue=0.0001, mag=1, count_penalty=10):
    live_samp = sample.copy()
    live_samp = pare(live_samp)
        
    cat_dummies = []
    drop_cols = []
    for column in live_samp.columns:
        if hasattr(live_samp[column], 'cat'):
            unify_weak_cats(live_samp, column, pvalue, mag, count_penalty)
            hot = pd.get_dummies(live_samp[column], prefix=column)
            cat_dummies.append(hot)
            drop_cols.append(column)
            
    live_samp.drop(drop_cols, axis=1)            
    
    one_hot_enc = pd.concat(cat_dummies, axis=1)
    
    model_samp = pd.concat([live_samp, one_hot_enc])
    feat_cols = model_samp.columns.drop(['AVG(AvgTone)'])
    
    train, test = train_test_split(live_samp, test_size=0.25, random_state=42)
    
    Y = train['AVG(AvgTone)']
    X = train[feat_cols]
    model.fit(X, Y)
    
    return train_score, test_score, model, model_samp, train, test

# can be used with pared and naive
def ICM_INT_sample(sample, country):
    cntry_sample = sample[sample['Actor1CountryCode'] == country].copy()
    cntry_sample['InternalEvent?'] = (cntry_sample['Actor2CountryCode'] == country).astype(int)
    cntry_sample = cntry_sample.drop(['Actor1CountryCode', 'Actor2CountryCode'], axis=1)
    return cntry_sample

# can be used with pared and naive
def ICM_SPEC_sample(sample, country):
    cntry_sample = sample[sample['Actor1CountryCode'] == country].copy()
    cntry_sample = cntry_sample.drop(['Actor1CountryCode'], axis=1)
    return cntry_sample   

# can only be used with naive
def ICM_INT_GINT_sample(sample, country):
    cntry_sample = sample[sample['Actor1CountryCode'] == country].copy()
    cntry_sample['InternalEvent?'] = (cntry_sample['Actor2CountryCode'] != country).astype(int)
    cntry_sample['Actor1AtHome?'] = (cntry_sample['Actor1Geo_CountryCode'] == country).astype(int)
    cntry_sample['Actor2AtActor1Home?'] = (cntry_sample['Actor2Geo_CountryCode'] == country).astype(int)
    cntry_sample = cntry_sample.drop(['Actor1CountryCode', 'Actor2CountryCode',
                                     'Actor1Geo_CountryCode', 'Actor2Geo_CountryCode']
                                     , axis=1)
    return cntry_sample    

# can only be used with naive
def ICM_INT_GSPEC_sample(sample, country):
    cntry_sample = sample[sample['Actor1CountryCode'] == country].copy()
    cntry_sample['InternalEvent?'] = (cntry_sample['Actor2CountryCode'] != country).astype(int)
    cntry_sample['Actor2AtActor1Home?'] = (cntry_sample['Actor2Geo_CountryCode'] == country).astype(int)
    cntry_sample = cntry_sample.drop(['Actor1CountryCode', 'Actor2CountryCode', 
                                      'Actor2Geo_CountryCode']
                                     , axis=1)
    return cntry_sample    

# can only be used with naive
def ICM_FULL_sample(sample, country):
    cntry_sample = sample[sample['Actor1CountryCode'] == country].copy()
    cntry_sample = cntry_sample.drop(['Actor1CountryCode'], axis=1)
    return cntry_sample    

# can only be used with naive
def ICM_PERSPEC_sample(sample, country):
    cntry_sample = sample[sample['Actor1CountryCode'] == country].copy()
    cntry_sample['Actor1AtHome?'] = (cntry_sample['Actor1Geo_CountryCode'] == country).astype(int)
    cntry_sample = cntry_sample.drop(['Actor1CountryCode', 'Actor1Geo_CountryCode']
                                     , axis=1)
    return cntry_sample

def country_info(sample, column, pvalue, mag, count_penalty):
    tones = sample['AVG(AvgTone)']
    cntry = sample[column]

    one_hot = pd.get_dummies(cntry)
    one_hot_tone = pd.concat([tones, one_hot], axis=1)

    avg_avgtone_mean = one_hot_tone['AVG(AvgTone)'].mean()

    country_info = []
    for column in one_hot.columns:
        temp = one_hot_tone[[column, 'AVG(AvgTone)']]
        country = temp[temp[column] == 1]
        columns_mean = None
        if len(country) < count_penalty: 
            column_means = [0, 0]
            country_info.append((column, 
                                 0, 
                                 0, 
                                 (temp[column].sum()),
                                 1))
        else:
            column_means = temp.groupby(column).mean()['AVG(AvgTone)']
            country_info.append((column, 
                                 column_means[0] - column_means[1], 
                                 np.absolute(column_means[0] - column_means[1]), 
                                 (temp[column].sum()),
                                 ttest_1samp(country, avg_avgtone_mean).pvalue[1]))

    cntry_spec = pd.DataFrame(country_info, columns=["Country", "AvgTone_diff", "AvgTone_mag", "Num", "p-value"])
    
    return cntry_spec
    
GDELT_columns = ["GLOBALEVENTID","SQLDATE","MonthYear","Year","FractionDate","Actor1Code","Actor1Name","Actor1CountryCode","Actor1KnownGroupCode","Actor1EthnicCode","Actor1Religion1Code","Actor1Religion2Code","Actor1Type1Code","Actor1Type2Code","Actor1Type3Code","Actor2Code","Actor2Name","Actor2CountryCode","Actor2KnownGroupCode","Actor2EthnicCode","Actor2Religion1Code","Actor2Religion2Code","Actor2Type1Code","Actor2Type2Code","Actor2Type3Code","IsRootEvent","EventCode","EventBaseCode","EventRootCode","QuadClass","GoldsteinScale","NumMentions","NumSources","NumArticles","AvgTone","Actor1Geo_Type","Actor1Geo_FullName","Actor1Geo_CountryCode","Actor1Geo_ADM1Code","Actor1Geo_Lat","Actor1Geo_Long","Actor1Geo_FeatureID","Actor2Geo_Type","Actor2Geo_FullName","Actor2Geo_CountryCode","Actor2Geo_ADM1Code","Actor2Geo_Lat","Actor2Geo_Long","Actor2Geo_FeatureID","ActionGeo_Type","ActionGeo_FullName","ActionGeo_CountryCode","ActionGeo_ADM1Code","ActionGeo_Lat","ActionGeo_Long","ActionGeo_FeatureID","DATEADDED","SOURCEURL"]
usecols = ['GLOBALEVENTID', 'SQLDATE', 'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode', 'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code', 'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code', 'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode', 'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code', 'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code', 'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode', 'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone', 'Actor1Geo_Type', 'Actor1Geo_FullName', 'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID', 'Actor2Geo_Type', 'Actor2Geo_FullName', 'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code', 'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID', 'ActionGeo_Type', 'ActionGeo_FullName', 'ActionGeo_CountryCode', 'ActionGeo_ADM1Code', 'ActionGeo_Lat', 'ActionGeo_Long', 'ActionGeo_FeatureID']
dtype_dict = {'GLOBALEVENTID': 'uint32', 
              'Actor1Code': 'category', 'Actor1Name': 'str', 'Actor1CountryCode': 'category', 'Actor1KnownGroupCode': 'category', 'Actor1EthnicCode': 'category', 'Actor1Religion1Code': 'category', 'Actor1Religion2Code': 'category', 'Actor1Type1Code': 'category', 'Actor1Type2Code': 'category', 'Actor1Type3Code': 'category',
              'Actor2Code': 'category', 'Actor2Name': 'str', 'Actor2CountryCode': 'category', 'Actor2KnownGroupCode': 'category', 'Actor2EthnicCode': 'category', 'Actor2Religion1Code': 'category', 'Actor2Religion2Code': 'category', 'Actor2Type1Code': 'category', 'Actor2Type2Code': 'category', 'Actor2Type3Code': 'category',
 'IsRootEvent': 'bool_',
 'EventCode': 'category', 'EventBaseCode': 'category', 'EventRootCode': 'category',
 'QuadClass': 'uint8', 'GoldsteinScale': 'float32', 'AvgTone': 'float32',
 'NumMentions': 'uint16', 'NumSources': 'uint16', 'NumArticles': 'uint16',
 'Actor1Geo_Type': 'float16', 'Actor1Geo_FullName': 'str', 'Actor1Geo_CountryCode': 'category', 'Actor1Geo_ADM1Code': 'category', 'Actor1Geo_Lat': 'float32', 'Actor1Geo_Long': 'float32', 'Actor1Geo_FeatureID': 'category', 
 'Actor2Geo_Type': 'float16', 'Actor2Geo_FullName': 'str', 'Actor2Geo_CountryCode': 'category', 'Actor2Geo_ADM1Code': 'category',  'Actor2Geo_Lat': 'float32', 'Actor2Geo_Long': 'float32', 'Actor2Geo_FeatureID': 'category',
 'ActionGeo_Type': 'float16', 'ActionGeo_FullName': 'str', 'ActionGeo_CountryCode': 'category', 'ActionGeo_ADM1Code': 'category', 'ActionGeo_Lat': 'float32', 'ActionGeo_Long': 'float32', 'ActionGeo_FeatureID': 'category'}
cameo_dict = {"01": "MAKE PUBLIC STATEMENT",
"010": "Make statement, not specified below",
"011": "Decline comment",
"012": "Make pessimistic comment",
"013": "Make optimistic comment",
"014": "Consider policy option",
"015": "Acknowledge or claim responsibility",
"016": "Deny responsibility",
"017": "Engage in symbolic act",
"018": "Make empathetic comment",
"019": "Express accord",
"02": "APPEAL",
"020": "Make an appeal or request, not specified below",
"021": "Appeal for material cooperation, not specified below",
"0211": "Appeal for economic cooperation",
"0212": "Appeal for military cooperation",
"0213": "Appeal for judicial cooperation",
"0214": "Appeal for intelligence",
"022": "Appeal for diplomatic cooperation (such as policy support)",
"023": "Appeal for aid, not specified below",
"0231": "Appeal for economic aid",
"0232": "Appeal for military aid",
"0233": "Appeal for humanitarian aid",
"0234": "Appeal for military protection or peacekeeping",
"024": "Appeal for political reform, not specified below",
"0241": "Appeal for change in leadership",
"0242": "Appeal for policy change",
"0243": "Appeal for rights",
"0244": "Appeal for change in institutions, regime",
"025": "Appeal to yield, not specified below",
"0251": "Appeal for easing of administrative sanctions",
"0252": "Appeal for easing of political dissent",
"0253": "Appeal for release of persons or property",
"0254": "Appeal for easing of economic sanctions, boycott, or embargo",
"0255": "Appeal for target to allow international involvement (non-mediation)",
"0256": "Appeal for de-escalation of military engagement",
"026": "Appeal to others to meet or negotiate",
"027": "Appeal to others to settle dispute",
"028": "Appeal to engage in or accept mediation",
"03": "EXPRESS INTENT TO COOPERATE",
"030": "Express intent to cooperate, not specified below",
"031": "Express intent to engage in material cooperation, not specified below",
"0311": "Express intent to cooperate economically",
"0312": "Express intent to cooperate militarily",
"0313": "Express intent to cooperate on judicial matters",
"0314": "Express intent to cooperate on intelligence",
"032": "Express intent to engage in diplomatic cooperation (such as policy support)",
"033": "Express intent to provide material aid, not specified below",
"0331": "Express intent to provide economic aid",
"0332": "Express intent to provide military aid",
"0333": "Express intent to provide humanitarian aid",
"0334": "Express intent to provide military protection or peacekeeping",
"034": "Express intent to institute political reform, not specified below",
"0341": "Express intent to change leadership",
"0342": "Express intent to change policy",
"0343": "Express intent to provide rights",
"0344": "Express intent to change institutions, regime",
"035": "Express intent to yield, not specified below",
"0351": "Express intent to ease administrative sanctions",
"0352": "Express intent to ease popular dissent",
"0353": "Express intent to release persons or property",
"0354": "Express intent to ease economic sanctions, boycott, or embargo",
"0355": "Express intent to allow international involvement (non-mediation)",
"0356": "Express intent to de-escalate military engagement",
"036": "Express intent to meet or negotiate",
"037": "Express intent to settle dispute",
"038": "Express intent to accept mediation",
"039": "Express intent to mediate",
"04": "CONSULT",
"040": "Consult, not specified below",
"041": "Discuss by telephone",
"042": "Make a visit",
"043": "Host a visit",
"044": "Meet at a 'third' location",
"045": "Mediate",
"046": "Engage in negotiation",
"05": "ENGAGE IN DIPLOMATIC COOPERATION",
"050": "Engage in diplomatic cooperation, not specified below",
"051": "Praise or endorse",
"052": "Defend verbally",
"053": "Rally support on behalf of",
"054": "Grant diplomatic recognition",
"055": "Apologize",
"056": "Forgive",
"057": "Sign formal agreement",
"06": "ENGAGE IN MATERIAL COOPERATION",
"060": "Engage in material cooperation, not specified below",
"061": "Cooperate economically",
"062": "Cooperate militarily",
"063": "Engage in judicial cooperation",
"064": "Share intelligence or information",
"07": "PROVIDE AID",
"070": "Provide aid, not specified below",
"071": "Provide economic aid",
"072": "Provide military aid",
"073": "Provide humanitarian aid",
"074": "Provide military protection or peacekeeping",
"075": "Grant asylum",
"08": "YIELD",
"080": "Yield, not specified below",
"081": "Ease administrative sanctions, not specified below",
"0811": "Ease restrictions on political freedoms",
"0812": "Ease ban on political parties or politicians",
"0813": "Ease curfew",
"0814": "Ease state of emergency or martial law",
"082": "Ease political dissent",
"083": "Accede to requests or demands for political reform, not specified below",
"0831": "Accede to demands for change in leadership",
"0832": "Accede to demands for change in policy",
"0833": "Accede to demands for rights",
"0834": "Accede to demands for change in institutions, regime",
"084": "Return, release, not specified below",
"0841": "Return, release person(s)",
"0842": "Return, release property",
"085": "Ease economic sanctions, boycott, embargo",
"086": "Allow international involvement, not specified below",
"0861": "Receive deployment of peacekeepers",
"0862": "Receive inspectors",
"0863": "Allow humanitarian access",
"087": "De-escalate military engagement",
"0871": "Declare truce, ceasefire",
"0872": "Ease military blockade",
"0873": "Demobilize armed forces",
"0874": "Retreat or surrender militarily",
"09": "INVESTIGATE",
"090": "Investigate, not specified below",
"091": "Investigate crime, corruption",
"092": "Investigate human rights abuses",
"093": "Investigate military action",
"094": "Investigate war crimes",
"10": "DEMAND",
"100": "Demand, not specified below",
"101": "Demand material cooperation, not specified below",
"1011": "Demand economic cooperation",
"1012": "Demand military cooperation",
"1013": "Demand judicial cooperation",
"1014": "Demand intelligence cooperation",
"102": "Demand diplomatic cooperation (such as policy support)",
"103": "Demand material aid, not specified below",
"1031": "Demand economic aid",
"1032": "Demand military aid",
"1033": "Demand humanitarian aid",
"1034": "Demand military protection or peacekeeping",
"104": "Demand political reform, not specified below",
"1041": "Demand change in leadership",
"1042": "Demand policy change",
"1043": "Demand rights",
"1044": "Demand change in institutions, regime",
"105": "Demand that target yields, not specified below",
"1051": "Demand easing of administrative sanctions",
"1052": "Demand easing of political dissent",
"1053": "Demand release of persons or property",
"1054": "Demand easing of economic sanctions, boycott, or embargo",
"1055": "Demand that target allows international involvement (non-mediation)",
"1056": "Demand de-escalation of military engagement",
"106": "Demand meeting, negotiation",
"107": "Demand settling of dispute",
"108": "Demand mediation",
"11": "DISAPPROVE",
"110": "Disapprove, not specified below",
"111": "Criticize or denounce",
"112": "Accuse, not specified below",
"1121": "Accuse of crime, corruption",
"1122": "Accuse of human rights abuses",
"1123": "Accuse of aggression",
"1124": "Accuse of war crimes",
"1125": "Accuse of espionage, treason",
"113": "Rally opposition against",
"114": "Complain ocially",
"115": "Bring lawsuit against",
"116": "Find guilty or liable (legally)",
"12": "REJECT",
"120": "Reject, not specified below",
"121": "Reject material cooperation",
"1211": "Reject economic cooperation",
"1212": "Reject military cooperation",
"122": "Reject request or demand for material aid, not specified below",
"1221": "Reject request for economic aid",
"1222": "Reject request for military aid",
"1223": "Reject request for humanitarian aid",
"1224": "Reject request for military protection or peacekeeping",
"123": "Reject request or demand for political reform, not specified below",
"1231": "Reject request for change in leadership",
"1232": "Reject request for policy change",
"1233": "Reject request for rights",
"1234": "Reject request for change in institutions, regime",
"124": "Refuse to yield, not specified below",
"1241": "Refuse to ease administrative sanctions",
"1242": "Refuse to ease popular dissent",
"1243": "Refuse to release persons or property",
"1244": "Refuse to ease economic sanctions, boycott, or embargo",
"1245": "Refuse to allow international involvement (non mediation)",
"1246": "Refuse to de-escalate military engagement",
"125": "Reject proposal to meet, discuss, or negotiate",
"126": "Reject mediation",
"127": "Reject plan, agreement to settle dispute",
"128": "Defy norms, law",
"129": "Veto",
"13": "THREATEN",
"130": "Threaten, not specified below",
"131": "Threaten non-force, not specified below",
"1311": "Threaten to reduce or stop aid",
"1312": "Threaten with sanctions, boycott, embargo",
"1313": "Threaten to reduce or break relations",
"132": "Threaten with administrative sanctions, not specified below",
"1321": "Threaten with restrictions on political freedoms",
"1322": "Threaten to ban political parties or politicians",
"1323": "Threaten to impose curfew",
"1324": "Threaten to impose state of emergency or martial law",
"133": "Threaten with political dissent, protest",
"134": "Threaten to halt negotiations",
"135": "Threaten to halt mediation",
"136": "Threaten to halt international involvement (non-mediation)",
"137": "Threaten with repression",
"138": "Threaten with military force, not specified below",
"1381": "Threaten blockade",
"1382": "Threaten occupation",
"1383": "Threaten unconventional violence",
"1384": "Threaten conventional attack",
"1385": "Threaten attack with WMD",
"139": "Give ultimatum",
"14": "PROTEST",
"140": "Engage in political dissent, not specified below",
"141": "Demonstrate or rally, not specified below",
"1411": "Demonstrate for leadership change",
"1412": "Demonstrate for policy change",
"1413": "Demonstrate for rights",
"1414": "Demonstrate for change in institutions, regime",
"142": "Conduct hunger strike, not specified below",
"1421": "Conduct hunger strike for leadership change",
"1422": "Conduct hunger strike for policy change",
"1423": "Conduct hunger strike for rights",
"1424": "Conduct hunger strike for change in institutions, regime",
"143": "Conduct strike or boycott, not specified below",
"1431": "Conduct strike or boycott for leadership change",
"1432": "Conduct strike or boycott for policy change",
"1433": "Conduct strike or boycott for rights",
"1434": "Conduct strike or boycott for change in institutions, regime",
"144": "Obstruct passage, block, not specified below",
"1441": "Obstruct passage to demand leadership change",
"1442": "Obstruct passage to demand policy change",
"1443": "Obstruct passage to demand rights",
"1444": "Obstruct passage to demand change in institutions, regime",
"145": "Protest violently, riot, not specified below",
"1451": "Engage in violent protest for leadership change",
"1452": "Engage in violent protest for policy change",
"1453": "Engage in violent protest for rights",
"1454": "Engage in violent protest for change in institutions, regime",
"15": "EXHIBIT FORCE POSTURE",
"150": "Demonstrate military or police power, not specified below",
"151": "Increase police alert status",
"152": "Increase military alert status",
"153": "Mobilize or increase police power",
"154": "Mobilize or increase armed forces",
"155": "Mobilize or increase cyber-forces",
"16": "REDUCE RELATIONS",
"160": "Reduce relations, not specified below",
"161": "Reduce or break diplomatic relations",
"162": "Reduce or stop material aid, not specified below",
"1621": "Reduce or stop economic assistance",
"1622": "Reduce or stop military assistance",
"1623": "Reduce or stop humanitarian assistance",
"163": "Impose embargo, boycott, or sanctions",
"164": "Halt negotiations",
"165": "Halt mediation",
"166": "Expel or withdraw, not specified below",
"1661": "Expel or withdraw peacekeepers",
"1662": "Expel or withdraw inspectors, observers",
"1663": "Expel or withdraw aid agencies",
"17": "COERCE",
"170": "Coerce, not specified below",
"171": "Seize or damage property, not specified below",
"1711": "Confiscate property",
"1712": "Destroy property",
"172": "Impose administrative sanctions, not specified below",
"1721": "Impose restrictions on political freedoms",
"1722": "Ban political parties or politicians",
"1723": "Impose curfew",
"1724": "Impose state of emergency or martial law",
"173": "Arrest, detain, or charge with legal action",
"174": "Expel or deport individuals",
"175": "Use tactics of violent repression",
"176": "Attack cybernetically",
"18": "ASSAULT",
"180": "Use unconventional violence, not specified below",
"181": "Abduct, hijack, or take hostage",
"182": "Physically assault, not specified below",
"1821": "Sexually assault",
"1822": "Torture",
"1823": "Kill by physical assault",
"183": "Conduct suicide, car, or other non-military bombing, not specified below",
"1831": "Carry out suicide bombing",
"1832": "Carry out vehicular bombing",
"1833": "Carry out roadside bombing",
"1834": "Carry out location bombing",
"184": "Use as human shield",
"185": "Attempt to assassinate",
"186": "Assassinate",
"19": "FIGHT",
"190": "Use conventional military force, not specified below",
"191": "Impose blockade, restrict movement",
"192": "Occupy territory",
"193": "Fight with small arms and light weapons",
"194": "Fight with artillery and tanks",
"195": "Employ aerial weapons, not specified below",
"1951": "Employ precision-guided aerial munitions",
"1952": "Employ remotely piloted aerial munitions",
"196": "Violate ceasefire",
"20": "USE UNCONVENTIONAL MASS VIOLENCE",
"200": "Use unconventional mass violence, not specified below",
"201": "Engage in mass expulsion",
"202": "Engage in mass killings",
"203": "Engage in ethnic cleansing",
"204": "Use weapons of mass destruction, not specified below",
"2041": "Use chemical, biological, or radiological weapons",
"2042": "Detonate nuclear weapons"}

def map_cameo_to_text(cameo_code):
    return cameo_dict[cameo_code]
#vmap_cameo_to_text = np.vectorize(map_cameo_to_text)

state_dict = {
        'USAK': 'Alaska',
        'USAL': 'Alabama',
        'USAR': 'Arkansas',
        'USAZ': 'Arizona',
        'USCA': 'California',
        'USCO': 'Colorado',
        'USCT': 'Connecticut',
        'USDC': 'District of Columbia',
        'USDE': 'Delaware',
        'USFL': 'Florida',
        'USGA': 'Georgia',    
    'USHI': 'Hawaii',
        'USIA': 'Iowa',
        'USID': 'Idaho',
        'USIL': 'Illinois',
        'USIN': 'Indiana',
        'USKS': 'Kansas',
        'USKY': 'Kentucky',
        'USLA': 'Louisiana',
        'USMA': 'Massachusetts',
        'USMD': 'Maryland',
        'USME': 'Maine',
        'USMI': 'Michigan',
        'USMN': 'Minnesota',
        'USMO': 'Missouri',
        'USMS': 'Mississippi',
        'USMT': 'Montana',
        'US': 'National',
        'USNC': 'North Carolina',
        'USND': 'North Dakota',
        'USNE': 'Nebraska',
        'USNH': 'New Hampshire',
        'USNJ': 'New Jersey',
        'USNM': 'New Mexico',
        'USNV': 'Nevada',
        'USNY': 'New York',
        'USOH': 'Ohio',
        'USOK': 'Oklahoma',
        'USOR': 'Oregon',
        'USPA': 'Pennsylvania',
        'USRI': 'Rhode Island',
        'USSC': 'South Carolina',
        'USSD': 'South Dakota',
        'USTN': 'Tennessee',
        'USTX': 'Texas',
        'USUT': 'Utah',
        'USVA': 'Virginia',
        'USVT': 'Vermont',
        'USWA': 'Washington',
        'USWI': 'Wisconsin',
        'USWV': 'West Virginia',
        'USWY': 'Wyoming',
        'USPR': 'Puerto Rico'
}

# Credit: https://www.dataquest.io/blog/pandas-big-data/ 
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: 
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 
    return usage_mb, "{:03.2f} MB".format(usage_mb)

def state_heat_map(data, vmin, vmax, title):
    fig, ax = plt.subplots(figsize=(14,7))
        
    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

    # Mercator projection, for Alaska and Hawaii
    m_ = Basemap(llcrnrlon=-190,llcrnrlat=20,urcrnrlon=-143,urcrnrlat=46,
    projection='merc',lat_ts=20) # do not change these numbers

    #%% ---------   draw state boundaries  ----------------------------------------
    ## data from U.S Census Bureau
    ## http://www.census.gov/geo/www/cob/st2000.html
    shp_info = m.readshapefile('st99_d00','states',drawbounds=True,
                               linewidth=0.45,color='gray')
    shp_info_ = m_.readshapefile('st99_d00','states',drawbounds=False)

    data_dict = data.to_dict()

    #%% -------- choose a color for each state based on population density. -------
    colors={}
    statenames=[]
    cmap = plt.cm.RdYlGn
    vmin = vmin; vmax = vmax # set range.
    norm = Normalize(vmin=vmin, vmax=vmax)
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['District of Columbia','Puerto Rico']:
            tone = data_dict[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            colors[statename] = cmap(np.sqrt((tone-vmin)/(vmax-vmin)))[:3]
        statenames.append(statename)

    #%% ---------  cycle through state names, color each one.  --------------------
    for nshape,seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['Puerto Rico', 'District of Columbia']:
            color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax.add_patch(poly)

    AREA_1 = 0.005  # exclude small Hawaiian islands that are smaller than AREA_1
    AREA_2 = AREA_1 * 30.0  # exclude Alaskan islands that are smaller than AREA_2
    AK_SCALE = 0.19  # scale down Alaska to show as a map inset
    HI_OFFSET_X = -1900000  # X coordinate offset amount to move Hawaii "beneath" Texas
    HI_OFFSET_Y = 250000    # similar to above: Y offset for Hawaii
    AK_OFFSET_X = -250000   # X offset for Alaska (These four values are obtained
    AK_OFFSET_Y = -750000 # via manual trial and error, thus changing them is not recommended.)

    for nshape, shapedict in enumerate(m_.states_info):  # plot Alaska and Hawaii as map insets
        if shapedict['NAME'] in ['Alaska', 'Hawaii']:
            seg = m_.states[int(shapedict['SHAPENUM'] - 1)]
            if shapedict['NAME'] == 'Hawaii' and float(shapedict['AREA']) > AREA_1:
                seg = [(x + HI_OFFSET_X, y + HI_OFFSET_Y) for x, y in seg]
                color = rgb2hex(colors[statenames[nshape]])
            elif shapedict['NAME'] == 'Alaska' and float(shapedict['AREA']) > AREA_2:
                seg = [(x*AK_SCALE + AK_OFFSET_X, y*AK_SCALE + AK_OFFSET_Y)\
                       for x, y in seg]
                color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor='gray', linewidth=.45)
            ax.add_patch(poly)

    ax.set_title(title)

    #%% ---------  Plot bounding boxes for Alaska and Hawaii insets  --------------
    light_gray = [0.8]*3  # define light gray color RGB
    x1,y1 = m_([-190,-183,-180,-180,-175,-171,-171],[29,29,26,26,26,22,20])
    x2,y2 = m_([-180,-180,-177],[26,23,20])  # these numbers are fine-tuned manually
    m_.plot(x1,y1,color=light_gray,linewidth=0.8)  # do not change them drastically
    m_.plot(x2,y2,color=light_gray,linewidth=0.8)

    #%% ---------   Show color bar  ---------------------------------------
    ax_c = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    cb = ColorbarBase(ax_c,cmap=cmap,norm=norm,orientation='vertical',
                      label=r'[average positivity]')

    plt.show();