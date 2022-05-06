python screen_risk_factors.py --dataset OneFlorida --goal anypasc --population positive 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anypasc-positive.txt
python screen_risk_factors.py --dataset OneFlorida --goal allpasc --population positive 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allpasc-positive.txt
python screen_risk_factors.py --dataset OneFlorida --goal anyorgan --population positive 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anyorgan-positive.txt
python screen_risk_factors.py --dataset OneFlorida --goal allorgan --population positive 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allorgan-positive.txt
python screen_risk_factors.py --dataset OneFlorida --goal anypasc --population negative 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anypasc-negative.txt
python screen_risk_factors.py --dataset OneFlorida --goal allpasc --population negative 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allpasc-negative.txt
python screen_risk_factors.py --dataset OneFlorida --goal anyorgan --population negative 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anyorgan-negative.txt
python screen_risk_factors.py --dataset OneFlorida --goal allorgan --population negative 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allorgan-negative.txt
python screen_risk_factors.py --dataset OneFlorida --goal anypasc --population all 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anypasc-all.txt
python screen_risk_factors.py --dataset OneFlorida --goal allpasc --population all 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allpasc-all.txt
python screen_risk_factors.py --dataset OneFlorida --goal anyorgan --population all 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anyorgan-all.txt
python screen_risk_factors.py --dataset OneFlorida --goal allorgan --population all 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allorgan-all.txt
python screen_risk_factors.py --dataset OneFlorida --goal anypasc --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anypasc-positive-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal allpasc --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allpasc-positive-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal anyorgan --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anyorgan-positive-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal allorgan --population positive --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allorgan-positive-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal anypasc --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anypasc-negative-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal allpasc --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allpasc-negative-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal anyorgan --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anyorgan-negative-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal allorgan --population negative --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allorgan-negative-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal anypasc --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anypasc-all-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal allpasc --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allpasc-all-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal anyorgan --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anyorgan-all-outpatient.txt
python screen_risk_factors.py --dataset OneFlorida --goal allorgan --population all --severity outpatient 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allorgan-all-outpatient.txt
