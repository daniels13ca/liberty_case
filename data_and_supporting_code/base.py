import time
import pandas as pd
import numpy as np


def generate_lead_base(n=20_000, seed=None):

    np.random.seed(seed if seed is not None else int(time.time()))

    # --- Step 1: Users ---
    phone_numbers = np.random.randint(60_000_000, 69_999_999, size=n)
    ages = (np.random.gamma(shape=2, scale=7, size=n) + 18).astype(int)
    ages = np.clip(ages, 18, 80)
    segments = np.random.choice(['A', 'B', 'C', 'D'], size=n, p=[0.10, 0.15, 0.45, 0.30])

    df = pd.DataFrame({
        'phone_number': phone_numbers,
        'age':          ages,
        'socio_economic_segment': segments
    })

    # --- Step 2: Mobility & Device ---
    mobility_categories = [
        'Home - Mostly in one cell',
        'Home - Work profile',
        'Home - Work - Third Site Profile',
        'High mobility - Multiple Sites'
    ]
    device_categories = [
        'iPhone high end',
        'iPhone mid end',
        'Android high end',
        'Android low end'
    ]

    def mobility_probs(age, segment):
        p = np.array([0.25, 0.35, 0.25, 0.15])
        if age > 60:
            p += [0.30, 0.05, -0.15, -0.20]
        if segment == 'A':
            p += [0.15, 0.10, 0.05, -0.30]
        elif segment == 'D':
            p += [-0.10, -0.05, 0.00, 0.15]
        p = np.clip(p, 0.01, None)
        return p / p.sum()

    def device_probs(age, segment):
        p = np.array([0.15, 0.20, 0.30, 0.35])
        if segment == 'A':
            p += [0.35, 0.15, 0.05, -0.55]
        elif segment == 'B':
            p += [0.10, 0.10, 0.10, -0.30]
        elif segment == 'D':
            p += [-0.08, -0.07, -0.05, 0.20]
        if age < 30:
            p += [0.05, 0.05, 0.08, -0.18]
        p = np.clip(p, 0.01, None)
        return p / p.sum()

    df['mobility_profile'] = [
        np.random.choice(mobility_categories, p=mobility_probs(a, s))
        for a, s in zip(df['age'], df['socio_economic_segment'])
    ]
    df['device'] = [
        np.random.choice(device_categories, p=device_probs(a, s))
        for a, s in zip(df['age'], df['socio_economic_segment'])
    ]

    # --- Step 3: DPI Usage ---
    total_usage_gb = np.round(np.clip(np.random.gamma(shape=2, scale=4, size=n) + 0.5, 0.5, 60), 2)
    # [YouTube, WhatsApp, Netflix, Disney+, Other OTT, News, Sports, Other]
    alpha = [3.5, 1.2, 1.8, 0.8, 1.0, 0.2, 0.4, 3.0]
    props = np.random.dirichlet(alpha, size=n)

    df['total_usage_gb']    = total_usage_gb
    df['youtube_gb']        = np.round(total_usage_gb * props[:, 0], 3)
    df['whatsapp_gb']       = np.round(total_usage_gb * props[:, 1], 3)
    df['netflix_gb']        = np.round(total_usage_gb * props[:, 2], 3)
    df['disney_gb']         = np.round(total_usage_gb * props[:, 3], 3)
    df['other_ott_gb']      = np.round(total_usage_gb * props[:, 4], 3)
    df['news_portals_gb']   = np.round(total_usage_gb * props[:, 5], 3)
    df['sports_portals_gb'] = np.round(total_usage_gb * props[:, 6], 3)

    return df