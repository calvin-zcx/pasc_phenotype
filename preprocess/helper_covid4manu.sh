python pre_cohort_manuscript.py --dataset COL 2>&1 | tee  log/pre_cohort_manuscript_NegNoCovidV2_COL.txt
python pre_cohort_manuscript.py --dataset WCM 2>&1 | tee  log/pre_cohort_manuscript_NegNoCovidV2_WCM.txt
python pre_cohort_manuscript.py --dataset NYU 2>&1 | tee  log/pre_cohort_manuscript_NegNoCovidV2_NYU.txt
python pre_cohort_manuscript.py --dataset MONTE 2>&1 | tee  log/pre_cohort_manuscript_NegNoCovidV2_MONTE.txt
python pre_cohort_manuscript.py --dataset MSHS 2>&1 | tee  log/pre_cohort_manuscript_NegNoCovidV2_MSHS.txt
