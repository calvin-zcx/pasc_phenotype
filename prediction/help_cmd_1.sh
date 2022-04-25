python screen_risk_factors.py --dataset INSIGHT --goal anypasc 2>&1 | tee  log/screen_risk_factors-insight-elix-anypasc.txt
python screen_risk_factors.py --dataset INSIGHT --goal allpasc 2>&1 | tee  log/screen_risk_factors-insight-elix-allpasc.txt
python screen_risk_factors.py --dataset INSIGHT --goal anyorgan 2>&1 | tee  log/screen_risk_factors-insight-elix-anyorgan.txt
python screen_risk_factors.py --dataset INSIGHT --goal allorgan 2>&1 | tee  log/screen_risk_factors-insight-elix-allorgan.txt
