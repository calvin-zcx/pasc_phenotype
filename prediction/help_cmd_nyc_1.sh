#python screen_risk_factors.py --dataset INSIGHT --goal anypasc 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc.txt
#python screen_risk_factors.py --dataset INSIGHT --goal allpasc 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc.txt
#python screen_risk_factors.py --dataset INSIGHT --goal anyorgan 2>&1 | tee  log/screen_risk_factors-insight-elix-anyorgan.txt
#python screen_risk_factors.py --dataset INSIGHT --goal allorgan 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypasc --population positive 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc-positive.txt
python screen_risk_factors.py --dataset INSIGHT --goal allpasc --population positive 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc-positive.txt
python screen_risk_factors.py --dataset INSIGHT --goal anyorgan --population positive 2>&1 | tee  log/screen_risk_factors-insight-elix-anyorgan-positive.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population positive 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-positive.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypasc --population negative 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc-negative.txt
python screen_risk_factors.py --dataset INSIGHT --goal allpasc --population negative 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc-negative.txt
python screen_risk_factors.py --dataset INSIGHT --goal anyorgan --population negative 2>&1 | tee  log/screen_risk_factors-insight-elix-anyorgan-negative.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population negative 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-negative.txt
python screen_risk_factors.py --dataset INSIGHT --goal anypasc --population all 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc-all.txt
python screen_risk_factors.py --dataset INSIGHT --goal allpasc --population all 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc-all.txt
python screen_risk_factors.py --dataset INSIGHT --goal anyorgan --population all 2>&1 | tee  log/screen_risk_factors-insight-elix-anyorgan-all.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan --population all 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan-all.txt
