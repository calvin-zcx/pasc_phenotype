# python screen_paxlovid_iptw_pcornet.py  --cohorttype norisklabdx --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisklabdx-anyfollowupdx-V2.txt
# python screen_paxlovid_iptw_pcornet.py  --cohorttype pregnantlabdx --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-pregnantlabdx-anyfollowupdx-V2.txt
# python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisknopreglabdx --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisknopreglabdx-anyfollowupdx-V2.txt
#
# python screen_paxlovid_iptw_pcornet_negctrl.py  --cohorttype atrisknopreg --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisknopreg-all-negctrl.txt
# python screen_paxlovid_iptw_pcornet_negctrl.py  --cohorttype norisk --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisk-all-negctrl.txt

python screen_paxlovid_iptw_pcornet_negctrl.py  --cohorttype norisk --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisk-all-negctrlV3.txt
python screen_paxlovid_iptw_pcornet_negctrl.py  --cohorttype atrisk --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisk-all-negctrlV3.txt

python screen_paxlovid_iptw_pcornet_boostrap.py  --cohorttype atrisk --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisk-all-boostrapV3.txt
python screen_paxlovid_iptw_pcornet_boostrap.py  --cohorttype norisk --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisk-all-boostrapV3.txt
