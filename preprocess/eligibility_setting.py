import sys

# for linux env.
sys.path.insert(0, '..')


# Configure baseline and follow up setting here
# 2022-01-27 updates: align with CDC query and our morning discussion

INDEX_AGE_MINIMUM = 20


def _is_in_baseline(event_time, index_time):
    # baseline: -18 months to -7 days prior to the index date
    # 2022-01-27 updates: align with CDC query and our morning discussion
    return -18*30 <= (event_time - index_time).days <= -7


def _is_in_followup(event_time, index_time):
    # follow-up: 1 month to 6 month after the index date
    # 2022-01-27 updates: align with CDC query and our morning discussion
    return 30 <= (event_time - index_time).days <= 30*6


def _is_in_acute(event_time, index_time):
    # not validated, just for debug
    return -7 < (event_time - index_time).days <= 14

