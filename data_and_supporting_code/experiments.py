import time
import pandas as pd
import numpy as np


def random_offer(df):

    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(int(time.time()))

    internet = rng.choice(['high_speed', 'low_speed'], size=n)
    df['offer_high_speed_internet'] = (internet == 'high_speed').astype(int)
    df['offer_low_speed_internet']  = (internet == 'low_speed').astype(int)

    tv = rng.choice(['basic_tv', 'premium_tv', 'none'], size=n)
    df['offer_basic_tv']   = (tv == 'basic_tv').astype(int)
    df['offer_premium_tv'] = (tv == 'premium_tv').astype(int)

    df['offer_disney_ott']        = rng.integers(0, 2, size=n)
    df['offer_paramount_ott']     = rng.integers(0, 2, size=n)
    df['offer_voice_line']        = rng.integers(0, 2, size=n)
    df['offer_wall_to_wall_wifi'] = rng.integers(0, 2, size=n)

    df['offer_description'] = _build_offer_description(df)

    return df


def fixed_offer(df):

    df = df.copy()

    df['offer_high_speed_internet'] = 0
    df['offer_low_speed_internet']  = 1
    df['offer_basic_tv']            = 1
    df['offer_premium_tv']          = 0
    df['offer_disney_ott']          = 0
    df['offer_paramount_ott']       = 0
    df['offer_voice_line']          = 0
    df['offer_wall_to_wall_wifi']   = 0

    df['offer_description'] = _build_offer_description(df)

    return df

_OFFER_LABELS = [
    ('offer_high_speed_internet', 'High Speed Internet'),
    ('offer_low_speed_internet',  'Low Speed Internet'),
    ('offer_basic_tv',            'Basic TV'),
    ('offer_premium_tv',          'Premium TV'),
    ('offer_disney_ott',          'Disney OTT'),
    ('offer_paramount_ott',       'Paramount OTT'),
    ('offer_voice_line',          'Voice Line'),
    ('offer_wall_to_wall_wifi',   'Wall to Wall Wifi'),
]

def _build_offer_description(df):
    return df.apply(
        lambda row: ' + '.join(label for col, label in _OFFER_LABELS if row[col] == 1),
        axis=1
    )