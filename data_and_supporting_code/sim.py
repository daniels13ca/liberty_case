import time
import pandas as pd
import numpy as np

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


def add_propensity_scores(df_in):

    df = df_in.copy()

    df['offer_description'] = _build_offer_description(df)

    n = len(df)
    rng = np.random.default_rng(int(time.time()))

    # --- Categorical encodings ---
    seg_map = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
    seg = df['socio_economic_segment'].map(seg_map).values.astype(float)
    low_inc = (seg == 0).astype(float)

    is_home_only  = (df['mobility_profile'] == 'Home - Mostly in one cell').astype(float).values
    is_high_mob   = (df['mobility_profile'] == 'High mobility - Multiple Sites').astype(float).values
    is_home_work  = (df['mobility_profile'] == 'Home - Work profile').astype(float).values
    is_home_work_third   = (df['mobility_profile'] == 'Home - Work - Third Site Profile').astype(float).values

    is_iphone_hi  = (df['device'] == 'iPhone high end').astype(float).values
    is_android_hi = (df['device'] == 'Android high end').astype(float).values
    is_hi_end     = np.clip(is_iphone_hi + is_android_hi, 0, 1)
    is_low_end    = (df['device'] == 'Android low end').astype(float).values

    # --- Normalized continuous features [0, 1] ---
    def n01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    age       = df['age'].values.astype(float)
    age_n     = n01(age)
    seg_n     = n01(seg)
    total_n   = n01(df['total_usage_gb'].values)
    inv_total_n   = n01(1/df['total_usage_gb'].values)
    youtube_n = n01(df['youtube_gb'].values)
    wapp_n    = n01(df['whatsapp_gb'].values)
    netflix_n = n01(df['netflix_gb'].values)
    disney_n  = n01(df['disney_gb'].values)
    ott_n     = n01(df['other_ott_gb'].values)
    news_n    = n01(df['news_portals_gb'].values)
    sports_n  = n01(df['sports_portals_gb'].values)

    def to_range(raw, lo, hi, noise_std=0.003):
        raw = raw + rng.normal(0, noise_std, size=n)
        mn, mx = raw.min(), raw.max()
        return np.round(lo + (raw - mn) / (mx - mn + 1e-9) * (hi - lo), 5)

    # 1. High speed internet 
    df['propensity_high_speed_internet'] = to_range(
        2.0 * total_n + 1.5 * seg_n + 1.5 * is_home_only + 1.5 * is_iphone_hi
        + 1.2 * disney_n + 1.2 * netflix_n + 1.0 * ott_n
        + 0.8 * youtube_n + 0.6 * is_android_hi
        - 1.5 * low_inc - 1.2 * news_n - 1.5 * is_high_mob,
        -0.01, 0.01, noise_std=0.01
    )

    # 2. Low speed internet 
    df['propensity_low_speed_internet'] = to_range(
        2.0 * low_inc + 1.5 * is_high_mob + 1.5 * is_low_end + 1.5 * wapp_n
        - 2.0 * is_iphone_hi - 2 * is_android_hi - 2 * is_home_only - 2 * is_home_work- 2 * is_home_work_third,
        -0.015, 0.01, noise_std=0.01
    )

    # 3. Basic TV 
    df['propensity_basic_tv'] = to_range(
        2.0 * low_inc + 1.5 * (1 - total_n)
        - 4.0 * (1 - age_n) - 1.5 * is_hi_end - 1.5 * seg_n
        - 1 * is_home_work  - 1 * is_home_work_third - 1.2 * sports_n - 1.2 * news_n,
        -0.007, 0.005, noise_std=0.005
    )

    # 4. Premium TV 
    df['propensity_premium_tv'] = to_range(
        2.0 * seg_n + 1.5 * is_home_only
        - 1.5 * disney_n - 1.2 * netflix_n - 1.0 * ott_n,
        -0.005, 0.005, noise_std=0.005
    )

    # 5. Disney OTT 
    df['propensity_disney_ott'] = to_range(
        2.0 * (1 - age_n) + 2.0 * disney_n + 1.5 * total_n + 1.2 * sports_n
        - 1.5 * news_n - 1.5 * age_n,
        -0.007, 0.007, noise_std=0.007
    )

    # 6. Paramount OTT  — inverted-U on age, peaks ~40
    mid_age_n = n01(1 - np.abs(age - 40) / 40)
    df['propensity_paramount_ott'] = to_range(
        1.5 * mid_age_n + 1.5 * netflix_n + 1.2 * ott_n + 1.0 * news_n
        - 1.2 * sports_n - 1.5 * age_n - 1.5 * (1 - age_n),
        -0.005, 0.005, noise_std=0.005
    )

    # 7. Voice line
    df['propensity_voice_line'] = to_range(
        3.0 * age_n - 2.0 * (1 - age_n),
        -0.005, 0.002, noise_std=0.002
    )

    # 8. Wall to wall wifi — home-focused, high-end affluent users, active data consumers
    is_home_1_or_2 = df['mobility_profile'].isin(
                        ['Home - Mostly in one cell', 'Home - Work profile']
                    ).astype(float).values
    is_seg_a      = (seg == 3).astype(float)
    age_penalty   = ((age < 25) | (age > 60)).astype(float)
    df['propensity_wall_to_wall_wifi'] = to_range(
        2.0 * total_n + 2.0 * is_seg_a + 1.5 * is_home_1_or_2 + 1.5 * is_hi_end
        + 1.2 * youtube_n + 1.0 * ott_n
        - 2.0 * low_inc - 2.0 * age_penalty - 1*inv_total_n,
        -0.007, 0.007, noise_std=0.007
    )

    return df


def offer_result(df):

    n = len(df)
    rng = np.random.default_rng(int(time.time()))

    pairs = [
        ('propensity_high_speed_internet', 'offer_high_speed_internet',   40,   31.61),
        ('propensity_low_speed_internet',  'offer_low_speed_internet',    27,   20.56),
        ('propensity_basic_tv',            'offer_basic_tv',  10,   4.5),
        ('propensity_premium_tv',          'offer_premium_tv',    23,   11.78),
        ('propensity_disney_ott',          'offer_disney_ott',    10,   5),
        ('propensity_paramount_ott',       'offer_paramount_ott', 8,    4),
        ('propensity_voice_line',          'offer_voice_line',    8,    5.56),
        ('propensity_wall_to_wall_wifi',   'offer_wall_to_wall_wifi', 4,    2.25),
    ]

    df['total_price']  = sum(df[offer]*price for _, offer,price, _ in pairs)
    df['total_margin']  = sum(df[offer]*margin for _, offer,_, margin in pairs)
    df['price_propencity'] = 0.3*(-np.exp(-1/df['total_price'])+0.97)
    df['total_propensity']  = sum(df[prop] * df[offer] for prop, offer, _ , _ in pairs) + df['price_propencity']

    df['random_acceptance'] = rng.uniform(0, 0.015, size=n)
    df['converted']         = df['total_propensity'] > df['random_acceptance']
    df['total_margin']      = df['total_margin'] * df['converted']

    offer_cols = [col for _, col, _, _ in pairs]
    output_cols = (
        ['phone_number']
        + offer_cols
        + ['offer_description', 'total_propensity', 'total_price','total_margin','random_acceptance', 'converted']
    )

    return df[output_cols]

_BASE_COLS = [
    'phone_number', 'age', 'socio_economic_segment', 'mobility_profile',
    'device', 'total_usage_gb', 'youtube_gb', 'whatsapp_gb', 'netflix_gb',
    'disney_gb', 'other_ott_gb', 'news_portals_gb', 'sports_portals_gb',
]

_OFFER_COLS = [col for col, _ in _OFFER_LABELS]


def _validate(df):
    missing = [c for c in _BASE_COLS + _OFFER_COLS if c not in df.columns]
    if missing:
        print("Dataframe does not include the correct columns or the schemas are incorrect")
        print(f"  Missing columns: {missing}")
        return False

    for col in _OFFER_COLS:
        if not df[col].isin([0, 1]).all():
            print("Dataframe does not include the correct columns or the schemas are incorrect")
            print(f"  Column '{col}' must contain only 0 or 1")
            return False

    if not ((df['offer_high_speed_internet'] + df['offer_low_speed_internet']) == 1).all():
        print("Dataframe does not include the correct columns or the schemas are incorrect")
        print("  Each row must have exactly one internet offer (high speed or low speed)")
        return False

    if not ((df['offer_basic_tv'] + df['offer_premium_tv']) <= 1).all():
        print("Dataframe does not include the correct columns or the schemas are incorrect")
        print("  Each row can have at most one TV offer (basic or premium)")
        return False

    return True


def sim(df):
    if not _validate(df):
        return None
    prop = add_propensity_scores(df)
    result = offer_result(prop)
    offer_cols = [col for col in result.columns if 'offer_' in col]
    out_cols = (
        ['phone_number']
        + offer_cols
        + ['offer_description', 'total_price','total_margin', 'converted']
    )
    return result[out_cols]



