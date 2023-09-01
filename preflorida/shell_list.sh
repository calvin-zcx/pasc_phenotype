mkdir log
#python pre_lab_4covid.py --dataset covid_database 2>&1 | tee  log/pre_lab_COL.txt
#python pre_lab_4covid.py --dataset main_database 2>&1 | tee  log/pre_lab_WCM.txt
#python pre_demo.py --dataset all 2>&1 | tee  log/pre_demo_allcombined.txt
#python pre_covid_lab.py --dataset all 2>&1 | tee  log/pre_covid_lab_allcombined.txt
#python pre_diagnosis.py --dataset all 2>&1 | tee  log/pre_diagnosis_all.txt
#python pre_medication.py --dataset all 2>&1 | tee  log/pre_medication_all.txt
#python pre_encounter.py --dataset all 2>&1 | tee  log/pre_encounter_all.txt
#python pre_procedure.py --dataset all 2>&1 | tee  log/pre_procedure_all.txt
#python pre_immun.py --dataset all 2>&1 | tee  log/pre_immun_all.txt
#python pre_death.py --dataset all 2>&1 | tee  log/pre_death_all.txt
python pre_vital.py --dataset all 2>&1 | tee  log/pre_vital_all.txt
python pre_cohort_manuscript.py --dataset all 2>&1 | tee  log/pre_cohort_manuscriptV2_all_oneflorida.txt
