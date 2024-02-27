 #python screen_paxlovid_iptw_pcornet.py  --cohorttype norisk --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisk-all-V2.txt
 python screen_paxlovid_iptw_pcornet.py  --cohorttype pregnant --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-pregnant-all-V3.txt
 python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisknopreg --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisknopreg-all-V2.txt
