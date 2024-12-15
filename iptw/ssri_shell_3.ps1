#python screen_ssri_iptw_pcornet.py  --exptype bupropion-base-180-0-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-bupropion-base-180-0-clean-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype bupropion-acute0-15-clean  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-bupropion-acute0-15-clean-mentalcov.txt

#python screen_ssri_iptw_pcornet.py  --exptype bupropion-base-180-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-bupropion-base-180-0-mentalcov.txt
#python screen_ssri_iptw_pcornet.py  --exptype bupropion-acute0-15  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-bupropion-acute0-15-mentalcov.txt

python screen_ssri_iptw_pcornetV2.py  --exptype ssri-base180-acutevsnot --severity anxiety 2>&1 | tee  log_ssri_v2/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-anxiety-CFSCVDDeath.txt
python screen_ssri_iptw_pcornetV2.py  --exptype ssri-base180-acutevsnot --severity SMI 2>&1 | tee  log_ssri_v2/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-SMI-CFSCVDDeath.txt
python screen_ssri_iptw_pcornetV2.py  --exptype ssri-base180-acutevsnot --severity nonSMI 2>&1 | tee  log_ssri_v2/screen_ssri_iptw_pcornet-ssri-base180-acutevsnot-nonSMI-CFSCVDDeath.txt

