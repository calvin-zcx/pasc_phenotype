python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population positive --severity all 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-positive-all-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population positive --severity all 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-positive-all-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-positive-outpatient-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-positive-outpatient-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population positive --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-positive-inpatienticu-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population positive --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-positive-inpatienticu-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypasc --population positive --severity all 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc-positive-all-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypasc --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc-positive-outpatient-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypasc --population positive --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc-positive-inpatienticu-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal allpasc --population positive --severity all 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc-positive-all-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal allpasc --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc-positive-outpatient-May17.txt
python screen_risk_factors.py --dataset INSIGHT --goal allpasc --population positive --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc-positive-inpatienticu-May17.txt

#python screen_risk_factors.py --dataset INSIGHT --goal allpasc --population positive --severity all 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc-positive.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population positive --severity inpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-positive-inpatient.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population positive --severity icu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-positive-icu.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population positive --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-positive-inpatienticu.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population positive --severity inpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-positive-inpatient.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population positive --severity icu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-positive-icu.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population negative --severity all 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-negative-all.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-negative-outpatient.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population negative --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-negative-inpatienticu.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population negative --severity inpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-negative-inpatient.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population negative --severity icu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-negative-icu.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population negative --severity all 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-negative-all.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-negative-outpatient.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population negative --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-negative-inpatienticu.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population negative --severity inpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-negative-inpatient.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population negative --severity icu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-negative-icu.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population all --severity all 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-all-all.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-all-outpatient.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population all --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-all-inpatienticu.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population all --severity inpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-all-inpatient.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascsevere --population all --severity icu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascsevere-all-icu.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population all --severity all 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-all-all.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-all-outpatient.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population all --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-all-inpatienticu.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population all --severity inpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-all-inpatient.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anypascmoderate --population all --severity icu 2>&1 | tee  log/screen_risk_factors-insight-elix-anypascmoderate-all-icu.txt