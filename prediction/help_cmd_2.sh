python screen_risk_factors.py --dataset OneFlorida --goal anypasc 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anypasc.txt
python screen_risk_factors.py --dataset OneFlorida --goal allpasc 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allpasc.txt
python screen_risk_factors.py --dataset OneFlorida --goal anyorgan 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-anyorgan.txt
python screen_risk_factors.py --dataset OneFlorida --goal allorgan 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix-allorgan.txt
