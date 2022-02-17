python pre_cohort_for_screen.py --dataset COL --positive_only 2>&1 | tee  log/pre_cohort_for_screen_NoFollowupEC_COL.txt
python pre_cohort_for_screen.py --dataset WCM --positive_only 2>&1 | tee  log/pre_cohort_for_screen_NoFollowupEC_WCM.txt
python pre_cohort_for_screen.py --dataset NYU --positive_only 2>&1 | tee  log/pre_cohort_for_screen_NoFollowupEC_NYU.txt
python pre_cohort_for_screen.py --dataset MONTE --positive_only 2>&1 | tee  log/pre_cohort_for_screen_NoFollowupEC_MONTE.txt
python pre_cohort_for_screen.py --dataset MSHS --positive_only 2>&1 | tee  log/pre_cohort_for_screen_NoFollowupEC_MSHS.txt