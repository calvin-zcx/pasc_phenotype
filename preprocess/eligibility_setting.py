import sys

# for linux env.
sys.path.insert(0, '..')


# Configure baseline and follow up setting here
# 2022-01-27 updates: align with CDC query and our morning discussion

INDEX_AGE_MINIMUM = 20
BASELINE_LEFT = -540
BASELINE_RIGHT = -7
FOLLOWUP_LEFT = 31
FOLLOWUP_RIGHT = 180

print('Adopted Eligibility Setting:')
print("...INDEX_AGE_MINIMUM:", INDEX_AGE_MINIMUM)
print("...BASELINE_LEFT:", BASELINE_LEFT)
print("...BASELINE_RIGHT:", BASELINE_RIGHT)
print("...FOLLOWUP_LEFT:", FOLLOWUP_LEFT)
print("...FOLLOWUP_RIGHT:", FOLLOWUP_RIGHT)


def _is_in_baseline(event_time, index_time):
    # baseline: -18 months to -7 days prior to the index date
    # 2022-01-27 updates: align with CDC query and our morning discussion
    # However, in CDC excel, they use 570 = 19*30 days for 18 months? we use 540 = 18*30days.
    return BASELINE_LEFT <= (event_time - index_time).days <= BASELINE_RIGHT


def _is_in_followup(event_time, index_time):
    # follow-up: 1 month to 6 month after the index date
    # 2022-01-27 updates: align with CDC query and our morning discussion
    return FOLLOWUP_LEFT <= (event_time - index_time).days <= FOLLOWUP_RIGHT


def _is_in_index_period(event_time, index_time):
    return -1 <= (event_time - index_time).days <= 16

