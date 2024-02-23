python screen_paxlovid_iptw_pcornet.py  --cohorttype norisklabdx --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisklabdx-all.txt
python screen_paxlovid_iptw_pcornet.py  --cohorttype norisklabdx --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisklabdx-anyfollowupdx.txt
python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisklabdx --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisklabdx-anyfollowupdx.txt

#python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisklabdx --severity 2022-03 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisklabdx-2022-03.txt
