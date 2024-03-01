# python screen_paxlovid_iptw_pcornet.py  --cohorttype norisklabdx --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisklabdx-all-V2.txt
# python screen_paxlovid_iptw_pcornet.py  --cohorttype pregnantlabdx --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-pregnantlabdx-all-V2.txt
# python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisknopreglabdx --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisknopreglabdx-all-V2.txt
#
# #python screen_paxlovid_iptw_pcornet.py  --cohorttype norisk --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisk-anyfollowupdx-V2.txt
# python screen_paxlovid_iptw_pcornet.py  --cohorttype pregnant --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-pregnant-anyfollowupdx-V2.txt
# python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisknopreg --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisknopreg-anyfollowupdx-V2.txt



#python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisklabdx --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisklabdx-all.txt
#python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisklabdx --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisklabdx-anyfollowupdx.txt

#python screen_paxlovid_iptw_pcornet.py  --cohorttype norisklabdx --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisklabdx-all.txt


# python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisknopreg --severity pax1stwave 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisknopreg-pax1stwave-V2.txt
# python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisknopreg --severity pax2ndwave 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisknopreg-pax2ndwave-V2.txt

python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisk --severity pax2ndwave 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisk-pax2ndwave-V3.txt
python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisk --severity pax1stwave 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisk-pax1stwave-V3.txt
