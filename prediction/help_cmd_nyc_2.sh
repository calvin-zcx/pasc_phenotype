#python screen_risk_factors.py --dataset OneFlorida --goal anypasc 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anypasc.txt
#python screen_risk_factors.py --dataset OneFlorida --goal allpasc 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allpasc.txt
#python screen_risk_factors.py --dataset OneFlorida --goal anyorgan 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anyorgan.txt
#python screen_risk_factors.py --dataset OneFlorida --goal allorgan 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allorgan.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypasc --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc-positive-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal allpasc --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc-positive-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal anyorgan --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anyorgan-positive-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-positive-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypasc --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc-negative-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal allpasc --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc-negative-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal anyorgan --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anyorgan-negative-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-negative-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypasc --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc-all-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal allpasc --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc-all-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal anyorgan --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-anyorgan-all-outpatient.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-all-outpatient.txt
