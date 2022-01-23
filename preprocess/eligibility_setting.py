import sys

# for linux env.
sys.path.insert(0, '..')


# Configure baseline and follow up setting here
# Mayb change later
INDEX_AGE_MINIMUM = 20


def _is_in_baseline(event_time, index_time):
    # baseline: -18 month to -1 month prior to the index date
    # return -540 <= (event_time - index_time).days <= -30
    return -30*12*3 <= (event_time - index_time).days <= -30


def _is_in_followup(event_time, index_time):
    # baseline: 1 month to 5 month after the index date
    return 30 <= (event_time - index_time).days <= 180


def _is_in_acute(event_time, index_time):
    # baseline: 1 month to 5 month after the index date
    return -14 <= (event_time - index_time).days <= 14

