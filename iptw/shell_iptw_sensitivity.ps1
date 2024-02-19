python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisklabdx --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisklabdx-all.txt
python screen_paxlovid_iptw_pcornet.py  --cohorttype norisklabdx --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisklabdx-all.txt
