python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population positive 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-positive.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population negative 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-negative.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population all 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-all.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population positive --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-positive-inpatienticu.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population negative --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-negative-inpatienticu.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population all --severity inpatienticu 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-all-inpatienticu.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-positive-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-negative-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-all-outpatient.txt

