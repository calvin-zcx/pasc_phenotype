import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd

# Configure baseline and follow-up setting here
# 2022-01-27 updates: align with CDC query and our morning discussion

INDEX_AGE_MINIMUM = 20
INDEX_AGE_MINIMUM_20 = 20
INDEX_AGE_MINIMUM_18 = 18

# BASELINE_LEFT = -540
# change at 2022-02-09 by Tom meeting
BASELINE_LEFT = -1095
BASELINE_RIGHT = -7

FOLLOWUP_LEFT = 31
FOLLOWUP_RIGHT = 180

INPATIENT_LEFT = -1
INPATIENT_RIGHT = 16

# change at 2022-02-22 0 --> -1
VENTILATION_LEFT = -1
VENTILATION_RIGHT = 16

COMMORBIDITY_LEFT = -1095
COMMORBIDITY_RIGHT = 0

BASELINE_MEDICATION_LEFT = -365
BASELINE_MEDICATION_RIGHT = -7

COVIDMED_LEFT = -14
COVIDMED_RIGHT = 14

BMI_LEFT = -365
BMI_RIGHT = 7

SMOKE_LEFT = -1095
SMOKE_RIGHT = 7

BASELINE_PREGNANCY_LEFT = -365
BASELINE_PREGNANCY_RIGHT = 0


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

print("...BASELINE_MEDICATION_LEFT:", BASELINE_MEDICATION_LEFT)
print("...BASELINE_MEDICATION_RIGHT:", BASELINE_MEDICATION_RIGHT)

print("...COVIDMED_LEFT:", COVIDMED_LEFT)
print("...COVIDMED_RIGHT:", COVIDMED_RIGHT)

print("...BMI_LEFT:", BMI_LEFT)
print("...BMI_RIGHT:", BMI_RIGHT)

print("...SMOKE_LEFT:", SMOKE_LEFT)
print("...SMOKE_RIGHT:", SMOKE_RIGHT)

print("...BASELINE_PREGNANCY_LEFT:", BASELINE_PREGNANCY_LEFT)
print("...BASELINE_PREGNANCY_RIGHT:", BASELINE_PREGNANCY_RIGHT)


def _is_in_baseline(event_time, index_time):
    # baseline: -18 months to -7 days prior to the index date
    # 2022-01-27 updates: align with CDC query and our morning discussion
    # However, in CDC excel, they use 570 = 19*30 days for 18 months? we use 540 = 18*30days.
    # [-3 years, -7]
    # handling error, e.g. pd.to_datetime('2022-01-01', errors='coerce'), pd.to_datetime('1700-01-01', errors='coerce')
    try:
        return BASELINE_LEFT <= (pd.to_datetime(event_time, errors='coerce') - pd.to_datetime(index_time,
                                                                                              errors='coerce')).days <= BASELINE_RIGHT
    except Exception as e:
        print('[ERROR:]', e, file=sys.stderr)
        print('event_time:', event_time, 'index_time:', index_time, file=sys.stderr)
        return False


def _is_in_followup(event_time, index_time):
    # follow-up: 1 month to 6 month after the index date
    # 2022-01-27 updates: align with CDC query and our morning discussion
    # [31, 180]
    try:
        return FOLLOWUP_LEFT <= (pd.to_datetime(event_time, errors='coerce') - pd.to_datetime(index_time,
                                                                                              errors='coerce')).days <= FOLLOWUP_RIGHT
    except Exception as e:
        print('[ERROR:]', e, file=sys.stderr)
        print('event_time:', event_time, 'index_time:', index_time, file=sys.stderr)
        return False


def _is_in_inpatient_period(event_time, index_time):
    # Diagnosis in an inpatient care setting within 1 day prior to 16 days after the index event
    # [-1, 16]
    try:
        return INPATIENT_LEFT <= (pd.to_datetime(event_time, errors='coerce') - pd.to_datetime(index_time,
                                                                                               errors='coerce')).days <= INPATIENT_RIGHT
    except Exception as e:
        print('[ERROR:]', e, file=sys.stderr)
        print('event_time:', event_time, 'index_time:', index_time, file=sys.stderr)
        return False


def _is_in_ventilation_period(event_time, index_time):
    # 3 year prior to baseline
    # [0, 16] --> [-1, 16]
    try:
        return VENTILATION_LEFT <= (pd.to_datetime(event_time, errors='coerce') - pd.to_datetime(index_time,
                                                                                                 errors='coerce')).days <= VENTILATION_RIGHT
    except Exception as e:
        print('[ERROR:]', e, file=sys.stderr)
        print('event_time:', event_time, 'index_time:', index_time, file=sys.stderr)
        return False


def _is_in_comorbidity_period(event_time, index_time):
    # 3 year prior to baseline
    # [-1095, 0]
    try:
        return COMMORBIDITY_LEFT <= (pd.to_datetime(event_time, errors='coerce') - pd.to_datetime(index_time,
                                                                                                  errors='coerce')).days <= COMMORBIDITY_RIGHT
    except Exception as e:
        print('[ERROR:]', e, file=sys.stderr)
        print('event_time:', event_time, 'index_time:', index_time, file=sys.stderr)
        return False


def _is_in_medication_baseline(event_time, index_time):
    # 1 year prior to baseline
    try:
        return BASELINE_MEDICATION_LEFT <= (pd.to_datetime(event_time, errors='coerce') - pd.to_datetime(index_time,
                                                                                                         errors='coerce')).days <= BASELINE_MEDICATION_RIGHT
    except Exception as e:
        print('[ERROR:]', e, file=sys.stderr)
        print('event_time:', event_time, 'index_time:', index_time, file=sys.stderr)
        return False


def _is_in_covid_medication(event_time, index_time):
    # 14 days before or after the index event
    try:
        return COVIDMED_LEFT <= (pd.to_datetime(event_time, errors='coerce') - pd.to_datetime(index_time,
                                                                                              errors='coerce')).days <= COVIDMED_RIGHT
    except Exception as e:
        print('[ERROR:]', e, file=sys.stderr)
        print('event_time:', event_time, 'index_time:', index_time, file=sys.stderr)
        return False


def _is_in_bmi_period(event_time, index_time):
    # -365 -- + 7
    try:
        return BMI_LEFT <= (pd.to_datetime(event_time, errors='coerce') - pd.to_datetime(index_time,
                                                                                         errors='coerce')).days <= BMI_RIGHT
    except Exception as e:
        print('[ERROR:]', e, file=sys.stderr)
        print('event_time:', event_time, 'index_time:', index_time, file=sys.stderr)
        return False


def _is_in_smoke_period(event_time, index_time):
    # -365 -- + 7
    try:
        return SMOKE_LEFT <= (pd.to_datetime(event_time, errors='coerce') - pd.to_datetime(index_time,
                                                                                       errors='coerce')).days <= SMOKE_RIGHT
    except Exception as e:
        print('[ERROR:]', e, file=sys.stderr)
        print('event_time:', event_time, 'index_time:', index_time, file=sys.stderr)
        return False


def _is_in_pregnancy_comorbidity_period(event_time, index_time):
    # 1 year prior to baseline
    # [-365, 0]
    try:
        return BASELINE_PREGNANCY_LEFT <= (pd.to_datetime(event_time, errors='coerce') - pd.to_datetime(index_time,
                                                                                                  errors='coerce')).days <= BASELINE_PREGNANCY_RIGHT
    except Exception as e:
        print('[ERROR:]', e, file=sys.stderr)
        print('event_time:', event_time, 'index_time:', index_time, file=sys.stderr)
        return False

