python screen_ssri_iptw_pcornet.py  --exptype snri-base-180-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-base-180-0-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype snri-acute0-15  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-acute0-7-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-base-180-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-base-180-0-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-acute0-15  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-acute0-15-mentalcov.txt

python screen_ssri_iptw_pcornet.py  --exptype snri-base-180-0-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-base-180-0-clean-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype snri-acute0-15-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-acute0-7-clean-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-base-180-0-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-base-180-0-clean-mentalcov.txt
python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-acute0-15-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-acute0-15-clean-mentalcov.txt


##python screen_ssri_iptw_pcornet.py  --exptype snri-acute0-7  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-acute0-7.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-acute0-7  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-acute0-7.txt
##python screen_ssri_iptw_pcornet.py  --exptype snri-base-120-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-snri-base-120-0.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssriVSsnri-base-120-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSsnri-base-120-0.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssri-acute0-15  --severity '2022-03' 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-acute0-15-sub202203.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssri-base-120-0  --severity '2022-03' 2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base-120-0-sub202203.txt
##python screen_ssri_iptw_pcornet.py  --exptype ssriVSwellbutrin-acute0-15  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssriVSwellbutrin-acute0-15.txt
