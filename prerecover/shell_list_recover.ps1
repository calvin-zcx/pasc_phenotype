#python pre_lab.py --dataset wcm 2>&1 | tee  log\pre_lab_wcm.txt
#python pre_demo.py --dataset wcm 2>&1 | tee  log\pre_demo_wcm.txt
#python pre_covid_lab.py --dataset wcm 2>&1 | tee  log\pre_covid_lab_wcm.txt
#python pre_diagnosis.py --dataset wcm 2>&1 | tee  log/pre_diagnosis_wcm.txt
#python pre_medication.py --dataset wcm 2>&1 | tee  log/pre_medication_wcm.txt
#python pre_encounter.py --dataset wcm 2>&1 | tee  log/pre_encounter_wcm.txt
#python pre_procedure.py --dataset wcm 2>&1 | tee  log/pre_procedure_wcm.txt
python pre_immun.py --dataset wcm 2>&1 | tee  log/pre_immun_wcm.txt
python pre_death.py --dataset wcm 2>&1 | tee  log/pre_death_wcm.txt
 python pre_vital.py --dataset wcm 2>&1 | tee  log/pre_vital_wcm.txt
# python pre_cohort_manuscript.py --dataset all 2>&1 | tee  log/pre_cohort_manuscriptV2_all_oneflorida.txt
