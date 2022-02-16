import sys
# for linux env.
sys.path.insert(0, '..')

# Configure baseline and follow up setting here
# 2022-01-27 updates: align with CDC query and our morning discussion

INDEX_AGE_MINIMUM = 20

# BASELINE_LEFT = -540
# change at 2022-02-09 by Tom meeting
BASELINE_LEFT = -1095
BASELINE_RIGHT = -7

FOLLOWUP_LEFT = 31
FOLLOWUP_RIGHT = 180

INPATIENT_LEFT = -1
INPATIENT_RIGHT = 16

VENTILATION_LEFT = 0
VENTILATION_RIGHT = 16

COMMORBIDITY_LEFT = -1095
COMMORBIDITY_RIGHT = 0

print('Adopted Eligibility Setting:')
print("...INDEX_AGE_MINIMUM:", INDEX_AGE_MINIMUM)

print("...BASELINE_LEFT:", BASELINE_LEFT)
print("...BASELINE_RIGHT:", BASELINE_RIGHT)

print("...FOLLOWUP_LEFT:", FOLLOWUP_LEFT)
print("...FOLLOWUP_RIGHT:", FOLLOWUP_RIGHT)

print("...INPATIENT_LEFT:", INPATIENT_LEFT)
print("...INPATIENT_RIGHT:", INPATIENT_RIGHT)

print("...VENTILATION_LEFT:", VENTILATION_LEFT)
print("...VENTILATION_RIGHT:", VENTILATION_RIGHT)

print("...COMMORBIDITY_LEFT:", COMMORBIDITY_LEFT)
print("...COMMORBIDITY_RIGHT:", COMMORBIDITY_RIGHT)


def _is_in_baseline(event_time, index_time):
    # baseline: -18 months to -7 days prior to the index date
    # 2022-01-27 updates: align with CDC query and our morning discussion
    # However, in CDC excel, they use 570 = 19*30 days for 18 months? we use 540 = 18*30days.
    # [-3 years, -7]
    return BASELINE_LEFT <= (event_time - index_time).days <= BASELINE_RIGHT


def _is_in_followup(event_time, index_time):
    # follow-up: 1 month to 6 month after the index date
    # 2022-01-27 updates: align with CDC query and our morning discussion
    # [31, 180]
    return FOLLOWUP_LEFT <= (event_time - index_time).days <= FOLLOWUP_RIGHT


def _is_in_inpatient_period(event_time, index_time):
    # Diagnosis in an inpatient care setting within 1 day prior to 16 days after the index event
    # [-1, 16]
    return INPATIENT_LEFT <= (event_time - index_time).days <= INPATIENT_RIGHT


def _is_in_ventilation_period(event_time, index_time):
    # 3 year prior to baseline
    # [0, 16]
    return VENTILATION_LEFT <= (event_time - index_time).days <= VENTILATION_RIGHT


def _is_in_comorbidity_period(event_time, index_time):
    # 3 year prior to baseline
    # [-1095, 0]
    return COMMORBIDITY_LEFT <= (event_time - index_time).days <= COMMORBIDITY_RIGHT
